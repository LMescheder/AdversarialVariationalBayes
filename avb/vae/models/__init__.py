import tensorflow as tf

from avb.vae.models import (
    full0,
    conv0, conv1, conv3
)

encoder_dict = {
    'full0': full0.encoder,
    'conv0': conv0.encoder,
    'conv1': conv1.encoder,
    'conv3': conv3.encoder,
}

def get_encoder(model_name, config, scope='encoder'):
    model_func = encoder_dict[model_name]

    return tf.make_template(
        scope, model_func, config=config
    )
