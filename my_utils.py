"""
Functions that use multiple times
"""

from torch import nn
import torch
import numpy as np
import math


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )


# function for est. collision time
def processObstacle(robot, obstacle):
    safeTime = []
    co_obstacle = []
    unco_obstacle = []
    unco_dist = []
    x1, y1, vx1, vy1, r1 = 0, 0, robot[4], robot[5], robot[3]
    for i in range(len(obstacle)):
        x2, y2, vx2, vy2, r2 = obstacle[i][6], obstacle[i][7], obstacle[i][8], obstacle[i][9], obstacle[i][10]
        a = (vx1 - vx2) ** 2 + (vy1 - vy2) ** 2
        b = 2 * (x1 - x2) * (vx1 - vx2) + 2 * (y1 - y2) * (vy1 - vy2)
        c = (x1 - x2) ** 2 + (y1 - y2) ** 2 - (r1 + r2) ** 2
        d = b ** 2 - 4 * a * c
        # 保证t>=0时，y>0,则不会相撞
        # 如果c<=0,则直接处于相撞的状态，这种情况在物理意义上不存在
        # c>0是必须的
        if c <= 0:
            unco_obstacle.append(obstacle[i])
            unco_dist.append(obstacle[i][11])  # norm(np.asarray(obstacle[i][0:2]) - np.asarray(robot[0:2]))
        else:
            if a == 0:  # b肯定也为0，c>0,这种情况不可能相撞
                unco_obstacle.append(obstacle[i])
                unco_dist.append(obstacle[i][11])
            else:
                if b > 0:  # 不会相撞
                    unco_obstacle.append(obstacle[i])
                    unco_dist.append(obstacle[i][11])
                else:
                    if d == 0:
                        t = -b / (2 * a)
                        co_obstacle.append(obstacle[i])
                        safeTime.append(t)
                    elif d > 0:
                        t = (-math.sqrt(d) - b) / (2 * a)
                        co_obstacle.append(obstacle[i])
                        safeTime.append(t)
                    else:
                        unco_obstacle.append(obstacle[i])
                        unco_dist.append(obstacle[i][11])
    return co_obstacle, safeTime, unco_obstacle, unco_dist


def reset_state(co_obstacle, safeTime, unco_obstacle, unco_dist):
    state = []
    if len(unco_obstacle) != 0:
        indexs = np.argsort(-np.asarray(unco_dist))
        for i in indexs:
            state.append(unco_obstacle[i])

    if len(co_obstacle) != 0:
        indexs = np.argsort(-np.asarray(safeTime))
        for i in indexs:
            state.append(co_obstacle[i])
    # print(len(state))
    return state


def process_state(self_state, humans_state):
    robot = self_state.cpu().numpy()  # state[0:6]6维
    obstacle = humans_state.cpu().numpy()
    co_obstacle, safeTime, unco_obstacle, unco_dist = processObstacle(robot, obstacle)
    state = reset_state(co_obstacle, safeTime, unco_obstacle, unco_dist)
    return state
