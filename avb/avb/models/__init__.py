import tensorflow as tf

from avb.avb.models import (
    conv0, conv1, conv0_ac, conv1_ac
)

encoder_dict = {
    'conv0': conv0.encoder,
    'conv1': conv1.encoder,
}

encoder_ac_dict = {
    'conv0_ac': conv0_ac.encoder,
    'conv1_ac': conv1_ac.encoder,
}

adversary_dict = {
    'conv0': conv0.adversary,
}

def get_encoder(model_name, config, scope='encoder'):
    # Use seperate dictionary to throw exception if wrong network is specified
    if config['is_ac']:
        model_func = encoder_ac_dict[model_name]
    else:
        model_func = encoder_dict[model_name]

    return tf.make_template(
        scope, model_func, config=config
    )


def get_adversary(model_name, config, scope='adversary'):
    model_func = adversary_dict[model_name]
    return tf.make_template(
        scope, model_func, config=config
    )
