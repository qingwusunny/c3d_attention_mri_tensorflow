use attention mechanism to AD classification
the data is MRI images from ADNI
the base network is C3D model


write_data.py 
write the MRI data to tfrecord

model_unit.py
the base unit of my model

my_c3d.py
the c3d model adapt to AD classification

my_attention.py
add attention mechanshim to c3d model

train_multi_gpu.py
train the model 
