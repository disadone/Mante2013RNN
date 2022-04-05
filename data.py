
# generate the dataset

import tensorflow as tf

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers


D_m=tfd.Uniform(-0.1875,0.1875)
D_c=tfd.Uniform(-0.1875,0.1875)
rho_m=tfd.Normal(loc=0.,scale=1.)
rho_c=tfd.Normal(loc=0.,scale=1.)

T=750

num_motion_trial=8000
num_color_trial=8000

def gen_trials(num_trial,u_cm,u_cc):
    assert(u_cm+u_cc==1 and u_cm*u_cc==0)
    data=[];target=[];info=[]
    for _ in range(num_trial):
        d_m=D_m.sample()
        d_c=D_c.sample()
        ans_m=tf.sign(d_m)
        ans_c=tf.sign(d_c)
        ans=u_cc*ans_c+u_cm*ans_m

        u_m=d_m+rho_m.sample(T)
        u_c=d_c+rho_c.sample(T)

        data.append(tf.stack([u_m,u_c,u_cm*tf.ones(T),u_cc*tf.ones(T)],axis=1))
        target.append(ans)
        info.append([d_m,d_c,ans_m,ans_c])
    return data,target,info
mdata,mtarget,minfo=gen_trials(num_motion_trial, u_cm=1, u_cc=0)
cdata,ctarget,cinfo=gen_trials(num_color_trial, u_cm=0, u_cc=1)

data=tf.concat([mdata,cdata],axis=0)
target=tf.concat([mtarget,ctarget],axis=0)
dataset=tf.data.Dataset.from_tensor_slices((data,target))



path='./data/mante2013_u'
tf.data.experimental.save(dataset, path)