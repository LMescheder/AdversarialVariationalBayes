import tensorflow as tf

from avb.auxvae.models import (
    conv0, conv1, full0
)

encoder_dict = {
    'full0': full0.encoder,
    'conv0': conv0.encoder,
    'conv1': conv1.encoder,
}


encoder_aux_dict = {
    'full0': full0.encoder_aux,
    'conv0': conv0.encoder_aux,
    'conv1': conv1.encoder_aux,
}

decoder_aux_dict = {
    'full0': full0.decoder_aux,
    'conv0': conv0.decoder_aux,
    'conv1': conv1.decoder_aux,
}



def get_encoder(model_name, config, scope='encoder'):
    model_func = encoder_dict[model_name]

    return tf.make_template(
        scope, model_func, config=config
    )


def get_encoder_aux(model_name, config, scope='encoder_aux'):
    model_func = encoder_aux_dict[model_name]

    return tf.make_template(
        scope, model_func, config=config
    )


def get_decoder_aux(model_name, config, scope='decoder_aux'):
    model_func = decoder_aux_dict[model_name]

    return tf.make_template(
        scope, model_func, config=config
    )
