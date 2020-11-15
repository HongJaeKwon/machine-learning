import pyautogui
import time
import cv2
import numpy as np
import collections
import tensorflow as tf
import keyboard
import threading
import random

state = collections.deque(maxlen = 2)
memory = collections.deque(maxlen=5000)
reward_discount_rate=0.99
input_shape=(256,128,2)
img_shape=(256,128)
resize_shape=(1,256,128,2)

# i = tf.keras.Input(shape=input_shape)
# out = tf.keras.layers.Lambda(lambda x : x/255.)(i)
# out = tf.keras.layers.Conv2D(4, 3, strides=1, padding='same', activation='relu')(out)
# out = tf.keras.layers.MaxPool2D()(out)
# out = tf.keras.layers.Conv2D(16, 3, strides=1, padding='same', activation='relu')(out)
# out = tf.keras.layers.MaxPool2D()(out)
# out = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='relu')(out)
# out = tf.keras.layers.MaxPool2D()(out)
# out = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', activation='relu')(out)
# out = tf.keras.layers.MaxPool2D()(out)
# out = tf.keras.layers.Flatten()(out)
# out = tf.keras.layers.Dense(128, activation='relu')(out)
# out = tf.keras.layers.Dense(64, activation='relu')(out)
# q = tf.keras.layers.Dense(2, activation='relu')(out)
# model = tf.keras.Model(inputs = [i], outputs = [q])


i = tf.keras.Input(shape=input_shape)
out = tf.keras.layers.Lambda(lambda x : x/255.)(i)
out = tf.keras.layers.Conv2D(4, 3, strides=1, padding='same', activation='relu')(out)
out = tf.keras.layers.MaxPool2D()(out)
out = tf.keras.layers.Conv2D(16, 3, strides=1, padding='same', activation='relu')(out)
out = tf.keras.layers.MaxPool2D()(out)
out = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='relu')(out)
out = tf.keras.layers.MaxPool2D()(out)
out = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', activation='relu')(out)
out = tf.keras.layers.MaxPool2D()(out)
out = tf.keras.layers.Flatten()(out)
out = tf.keras.layers.Dense(128, activation='relu')(out)
out = tf.keras.layers.Dense(64, activation='relu')(out)
q = tf.keras.layers.Dense(2, activation='relu')(out)
t_model = tf.keras.Model(inputs = [i], outputs = [q])

model=tf.keras.models.load_model("hjk_dino_model_dqn2_7500.h5")
model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(0.001))
t_model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(0.001))
t_model.set_weights(model.get_weights())
start_e = 7500
model.summary()


def img_show():

    global state
    while True:
        if(len(state)==2):
            res = cv2.bitwise_not(state[1])
            cv2.imshow("HJK T-REX", res)
            cv2.waitKey(1)

def is_done():
    img = pyautogui.screenshot(region=(362, 326, 350, 20))
    img = np.array(img)
    if np.sum(img) > 5000000:
        return False
    else:
        return True

def t_rex_reset():
    global state
    time.sleep(2)
    for i in range(2):
        img = pyautogui.screenshot(region=(50, 100, 1000, 500))
        img = np.array(img)
        img = cv2.resize(img,img_shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        state.append(img)
    pyautogui.click(250, 500)
    res=cv2.merge(np.array(state))
    res=cv2.bitwise_not(res)
    return res

def t_rex_act(act):
    global state

    if act==0:
        pyautogui.press('up')
        # pass
    if act==1:
        pyautogui.press('down')
        # pass
    done = is_done()
    if not done:
        r = 1
    else:
        r = 0
    img = pyautogui.screenshot(region=(50, 100, 1000, 500))
    img = np.array(img)
    img = cv2.resize(img, img_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    state.append(img)
    res = cv2.merge(np.array(state))
    res = cv2.bitwise_not(res)
    return (res, r, done)

def obstacle():
    img = pyautogui.screenshot(region=(50+140, 300+100, 200, 100))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return np.sum(img)

total_reward_list=[]

t = threading.Thread(target=img_show)
t.daemon=True
t.start()

batch_size = 128
grad_t=[]
count=0
epsilon= 0.9
epsilon_min = 0.01
epsilon_decay = epsilon_min / epsilon
epsilon_decay = epsilon_decay ** (1. / float(5000))
train_count=0
target_update_count = 30

# while True:
#     print(pyautogui.position())
#     time.sleep(1)


for e in range(start_e):
    if (epsilon > epsilon_min):
        epsilon *= epsilon_decay
#
# for e in range(1000):
#     s = t_rex_reset()
#     done = False
#     while not done:
#         o=obstacle()
#         print(o)
#         if(o<5000000):
#             action=0
#         else:
#             action=1
#         n_s, r, done = t_rex_act(action)

for e in range(start_e, 10000):
    s = t_rex_reset()
    s = np.reshape(s, resize_shape)
    done = False
    total_reward = 0
    while not done:
        # s_time=time.time()
        # count = count + 1
        if e < 0:
            if keyboard.is_pressed('a'):
                action = 1
            else:
                action = 0
        else:
            q = model.predict(s)
            action=np.argmax(q[0])
            if (np.random.rand()) < epsilon:
                action = np.random.choice(2)
        n_s,r,done = t_rex_act(action)
        n_s = np.reshape(n_s, resize_shape)
        i = (s, action, r/10., n_s, done)
        memory.append(i)
        s = n_s
        total_reward = total_reward + r
        # if(count==10):
        #     res_1 = cv2.bitwise_not(state[0])
        #     cv2.imshow("1", res_1)
        #     res_2 = cv2.bitwise_not(state[1])
        #     cv2.imshow("2", res_2)
        #     cv2.waitKey(0)
        # print(time.time()-s_time)
    if(len(memory)>=batch_size*3):
        print("trained")
        sample = random.sample(memory, batch_size)
        state_batch = []
        q_val_batch = []
        for _state, _action, _reward, _next_state, _done in sample:
            q_val = model.predict(_state)
            target_q_val = _reward + reward_discount_rate * np.max(t_model.predict(_next_state)[0])

            if _done:
                q_val[0][_action] = _reward
            else:
                q_val[0][_action] = target_q_val

            state_batch.append(_state[0])
            q_val_batch.append(q_val[0])

            # 학습하고 타겟모델을 DQN모델로 업데이트 하고, 입실론 값을 줄임
        model.train_on_batch(np.array(state_batch), np.array(q_val_batch))
        if (epsilon > epsilon_min):
            epsilon *= epsilon_decay
        train_count = train_count + 1

        if train_count % target_update_count == 0:
            t_model.set_weights(model.get_weights())

    print(e, total_reward)
    total_reward_list.append(total_reward)

    if((e)%100==0):
        model.save("hjk_dino_model_dqn2_{}.h5".format(str(e)))
