import tensorflow as tf

from avb.avb.models_ac import conv0

encoder_dict = {
    'conv0_ac': conv0.encoder,
}

def get_encoder_ac(model_name, config, scope='encoder'):
    model_func = decoder_dict[model_name]
    return tf.make_template(
        scope, model_func, config
    )
