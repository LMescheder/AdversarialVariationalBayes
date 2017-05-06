import tensorflow as tf

from avb.avb.models import (
    conv0, conv1
)

encoder_dict = {
    'conv0': conv0.encoder,
    'conv1': conv1.encoder,
}

adversary_dict = {
    'conv0': conv0.adversary,
}

def get_encoder(model_name, config, scope='encoder'):
    model_func = encoder_dict[model_name]
    return tf.make_template(
        scope, model_func, config=config
    )


def get_adversary(model_name, config, scope='adversary'):
    model_func = adversary_dict[model_name]
    return tf.make_template(
        scope, model_func, config=config
    )
