import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Add,
    Dense,
    Flatten,
    Input,
    Lambda,
    Multiply,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K

tf.compat.v1.disable_v2_behavior()


MAKE_DETERMINISTIC = True


def build_vae(
    latent_dim,
    n_tokens=4,
    seq_length=33,
    enc1_units=50,
    eps_std=1.0,
):
    """Returns a compiled VAE model"""
    model = SimpleVAE(
        input_shape=(
            seq_length,
            n_tokens,
        ),
        latent_dim=latent_dim,
    )

    # set encoder layers:
    model.encoderLayers_ = [
        Dense(units=enc1_units, activation="elu", name="e2"),
    ]

    # set decoder layers:
    model.decoderLayers_ = [
        Dense(units=enc1_units, activation="elu", name="d1"),
        Dense(units=n_tokens * seq_length, name="d3"),
        Reshape((seq_length, n_tokens), name="d4"),
        Dense(units=n_tokens, activation="softmax", name="d5"),
    ]

    # build models:
    kl_scale = K.variable(1.0)
    model.build_encoder()
    model.build_decoder(decode_activation="softmax")
    model.build_vae(epsilon_std=eps_std, kl_scale=kl_scale)

    losses = [summed_categorical_crossentropy, identity_loss]

    model.compile(optimizer="adam", loss=losses)

    return model


def summed_categorical_crossentropy(y_true, y_pred):
    """Negative log likelihood of categorical distribution"""
    return K.sum(K.categorical_crossentropy(y_true, y_pred), axis=-1)


def identity_loss(y_true, y_pred):
    """Returns the predictions"""
    return y_pred


class BaseVAE(object):
    """Base class for Variational Autoencoders implemented in Keras

    The class is designed to connect user-specified encoder and decoder
    models via a Model representing the latent space

    """

    def __init__(self, input_shape, latent_dim, *args, **kwargs):
        self.latentDim_ = latent_dim
        self.inputShape_ = input_shape

        self.encoder_ = None
        self.decoder_ = None

        self.vae_ = None

    def build_encoder(self, *args, **kwargs):
        """Build the encoder network as a keras Model

        The encoder Model must ouput the mean and log variance of
        the latent space embeddings. I.e. this model must output
        mu and Sigma of the latent space distribution:

                    q(z|x) = N(z| mu(x), Sigma(x))

        Sets the value of self.encoder_ to a keras Model

        """

        raise NotImplementedError

    def build_decoder(self, *args, **kwargs):
        """Build the decoder network as a keras Model

        The input to the decoder must have the same shape as the latent
        space and the output must have the same shape as the input to
        the encoder.

        Sets the value of self.decoder_ to a keras Model

        """

        raise NotImplementedError

    def _build_latent_vars(self, mu_z, log_var_z, epsilon_std=1.0, kl_scale=1.0):
        """Build keras variables representing the latent space

        First, calculate the KL divergence from the input mean and log variance
        and add this to the model loss via a KLDivergenceLayer. Then sample an epsilon
        and perform a location-scale transformation to obtain the latent embedding, z.

        Args:
            epsilon_std: standard deviation of p(epsilon)
            kl_scale: weight of KL divergence loss

        Returns
            Variables representing z and epsilon

        """

        # mu_z, log_var_z, kl_batch  = KLDivergenceLayer()([mu_z, log_var_z], scale=kl_scale)
        lmda_func = lambda inputs: -0.5 * K.sum(
            1 + inputs[1] - K.square(inputs[0]) - K.exp(inputs[1]), axis=1
        )

        kl_batch = Lambda(lmda_func, name="kl_calc")([mu_z, log_var_z])
        kl_batch = Reshape((1,), name="kl_reshape")(kl_batch)

        # get standard deviation from log variance:
        sigma_z = Lambda(lambda lv: K.exp(0.5 * lv))(log_var_z)

        if MAKE_DETERMINISTIC:
            eps = Input(tensor=K.zeros_like(mu_z))
        else:
            # re-parametrization trick ( z = mu_z + eps * sigma_z)
            eps = Input(
                tensor=K.random_normal(
                    stddev=epsilon_std, shape=(K.shape(mu_z)[0], self.latentDim_)
                )
            )

        eps_z = Multiply()([sigma_z, eps])  # scale by epsilon sample
        z = Add()([mu_z, eps_z])

        return z, eps, kl_batch

    def _get_decoder_input(self, z, enc_in):
        return z

    def build_vae(self, epsilon_std=1.0, kl_scale=1.0):
        """Build the VAE

        Sets the value of self.vae_ to a keras model

        Args:
            epsilon_std (float): standard deviation of distribution used to sample z via the reparameratization trick
            kl_scale (float or keras.backend.Variable): weight of KL divergence loss

        """

        if self.encoder_ is None:
            raise TypeError("Encoder must be built before calling build_vae")
        if self.decoder_ is None:
            raise TypeError("Decoder must be built before calling build_vae")

        enc_in = self.encoder_.inputs
        mu_z, log_var_z = self.encoder_.outputs
        z, eps, kl_batch = self._build_latent_vars(
            mu_z, log_var_z, epsilon_std=epsilon_std, kl_scale=kl_scale
        )
        dec_in = self._get_decoder_input(z, enc_in)
        x_pred = self.decoder_(dec_in)
        self.vae_ = Model(
            inputs=enc_in + [eps], outputs=[x_pred, kl_batch], name="vae_base"
        )

    def plot_model(self, *args, **kwargs):
        keras.utils.plot_model(self.vae_, *args, **kwargs)

    def compile(self, *args, **kwargs):
        self.vae_.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        self.vae_.fit(*args, **kwargs)

    def save_all_weights(self, prefix):
        encoder_file = prefix + "_encoder.h5"
        decoder_file = prefix + "_decoder.h5"
        vae_file = prefix + "_vae.h5"

        self.encoder_.save_weights(encoder_file)
        self.decoder_.save_weights(decoder_file)
        self.vae_.save_weights(vae_file)

    def load_all_weights(self, prefix):
        encoder_file = prefix + "_encoder.h5"
        decoder_file = prefix + "_decoder.h5"
        vae_file = prefix + "_vae.h5"

        self.encoder_.load_weights(encoder_file)
        self.decoder_.load_weights(decoder_file)
        self.vae_.load_weights(vae_file)


class SimpleVAE(BaseVAE):
    """Basic VAE where the encoder and decoder can be constructed from lists of layers"""

    def __init__(self, input_shape, latent_dim, flatten=True, *args, **kwargs):
        super(SimpleVAE, self).__init__(
            input_shape=input_shape, latent_dim=latent_dim, *args, **kwargs
        )
        self.flatten_ = flatten
        self.encoderLayers_ = []
        self.decoderLayers_ = []

    def add_encoder_layer(self, layer):
        """Append a keras Layer to self.encoderLayers_"""
        self.encoderLayers_.append(layer)

    def add_decoder_layer(self, layer):
        """Append a keras Layer to self.decoderLayers_"""
        self.decoderLayers_.append(layer)

    def _build_encoder_inputs(self):
        """BUILD (as opposed to get) the encoder inputs"""
        x = Input(shape=self.inputShape_)
        return [x]

    def _build_decoder_inputs(self):
        z = Input(shape=(self.latentDim_,))
        return z

    def _edit_encoder_inputs(self, enc_in):
        if self.flatten_:
            h = Flatten()(enc_in[0])
        else:
            h = enc_in[0]
        return h

    def _edit_decoder_inputs(self, dec_in):
        return dec_in

    def build_encoder(self):
        """Construct the encoder from list of layers

        After the final layer in self.encoderLayers_, two Dense layers
        are applied to output mu_z and log_var_z

        """

        if len(self.encoderLayers_) == 0:
            raise ValueError("Must add at least one encoder hidden layer")

        enc_in = self._build_encoder_inputs()
        h = self._edit_encoder_inputs(enc_in)
        for hid in self.encoderLayers_:
            h = hid(h)

        mu_z = Dense(self.latentDim_, name="mu_z")(h)
        log_var_z = Dense(self.latentDim_, name="log_var_z")(h)

        self.encoder_ = Model(inputs=enc_in, outputs=[mu_z, log_var_z], name="encoder")

    def build_decoder(self, decode_activation):
        """Construct the decoder from list of layers

        After the final layer in self.decoderLayers_, a Dense layer is
        applied to output the final reconstruction

        Args:
            decode_activation: activation of the final decoding layer

        """

        if len(self.decoderLayers_) == 0:
            raise ValueError("Must add at least one decoder hidden layer")

        dec_in = self._build_decoder_inputs()
        h = self._edit_decoder_inputs(dec_in)
        for hid in self.decoderLayers_:
            h = hid(h)

        x_pred = h
        self.decoder_ = Model(inputs=dec_in, outputs=x_pred, name="decoder")


def build_vae(
    latent_dim,
    n_tokens=4,
    seq_length=33,
    enc1_units=50,
    eps_std=1.0,
):
    """Returns a compiled VAE model"""
    model = SimpleVAE(
        input_shape=(
            seq_length,
            n_tokens,
        ),
        latent_dim=latent_dim,
    )

    # set encoder layers:
    model.encoderLayers_ = [
        Dense(units=enc1_units, activation="elu", name="e2"),
    ]

    # set decoder layers:
    model.decoderLayers_ = [
        Dense(units=enc1_units, activation="elu", name="d1"),
        Dense(units=n_tokens * seq_length, name="d3"),
        Reshape((seq_length, n_tokens), name="d4"),
        Dense(units=n_tokens, activation="softmax", name="d5"),
    ]

    # build models:
    kl_scale = K.variable(1.0)
    model.build_encoder()
    model.build_decoder(decode_activation="softmax")
    model.build_vae(epsilon_std=eps_std, kl_scale=kl_scale)

    losses = [summed_categorical_crossentropy, identity_loss]

    model.compile(optimizer="adam", loss=losses)

    return model
