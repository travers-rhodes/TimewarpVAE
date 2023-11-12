import timewarp_lib.scalar_timewarpers as tim
import timewarp_lib.encoders as et
import timewarp_lib.decoders as dt
import timewarp_lib.vector_timewarpers as vtw
import timewarp_lib.vae_template as vl
import timewarp_lib.rate_invariant_autoencoder_template as ria

def parse_encoder(**kwargs):
  if kwargs["encoder_name"] == "transformer_encoder":
    encoder = et.TransformerEncoder(**kwargs)
  elif kwargs["encoder_name"] == "self_attention_transformer_encoder":
    encoder = et.SelfAttentionTransformerEncoder(**kwargs)
  elif kwargs["encoder_name"] == "convolutional_encoder":
    encoder = et.OneDConvEncoder(**kwargs)
  else:
    requested = kwargs["encoder_name"]
    raise Exception(f"{requested} encoder is not known")
  return encoder


## parse all the kwargs
def parse_arguments(**kwargs):
  use_rate_invariant_autoencoder = kwargs.get("use_rate_invariant_autoencoder",False)
  if use_rate_invariant_autoencoder:
    added_latent_dim_for_encoder = kwargs["ria_T"]
    encoder_kwargs = kwargs.copy()
    encoder_kwargs["latent_dim"] += added_latent_dim_for_encoder
  else:
    encoder_kwargs = kwargs

  if kwargs["scalar_timewarper_name"] == "identity_scalar_timewarper":
    scalar_timewarper = tim.IdentityScalarTimewarper(**kwargs)
  elif kwargs["scalar_timewarper_name"] == "modeled_scalar_timewarper":
    scalar_timewarper = tim.ModeledParameterScalarTimewarper(**kwargs)
  else:
    requested = kwargs["scalar_timewarper_name"]
    raise Exception(f"{requested} scalar timewarper is not known")

  encoder = parse_encoder(**encoder_kwargs)

  if kwargs["decoder_name"] == "functional_decoder":
    decoder = dt.FunctionStyleDecoder(**kwargs)
  elif kwargs["decoder_name"] == "convolutional_decoder":
    decoder = dt.OneDConvDecoder(**kwargs)
  elif kwargs["decoder_name"] == "convolutional_decoder_upsampling":
    decoder = dt.OneDConvDecoderUpsampling(**kwargs)
  elif kwargs["decoder_name"] == "functional_decoder_complicated":
    decoder = dt.ComplicatedFunctionStyleDecoder(**kwargs)
  else:
    requested = kwargs["decoder_name"]
    raise Exception(f"{requested} decoder is not known")

  if kwargs["vector_timewarper_name"] == "dtw_vector_timewarper":
    vector_timewarper = vtw.DTWVectorTimewarper(**kwargs)
  elif kwargs["vector_timewarper_name"] == "linear_dtw_vector_timewarper":
    vector_timewarper = vtw.LinearDTWVectorTimewarper(**kwargs)
  elif kwargs["vector_timewarper_name"] == "identity_vector_timewarper":
    vector_timewarper = vtw.IdentityVectorTimewarper(**kwargs)
  else:
    requested = kwargs["vector_timewarper_name"]
    raise Exception(f"{requested} vector timewarper is not known")

  if use_rate_invariant_autoencoder:
    hi = ria.RateInvariantAutoencoder(encoder=encoder,decoder=decoder, **kwargs)
  else:
    hi = vl.VAE(encoder=encoder,decoder=decoder,scalar_timewarper=scalar_timewarper)
  return hi,vector_timewarper
  
