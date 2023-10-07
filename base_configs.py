import numpy as np

NUM_CHANNELS = 2

convup_no_dtw = dict(
     ###########################
     #####ScalarTimeWarping#####
     ###########################
     ###IdentityScalarTimewarper###
     scalar_timewarper_name = "identity_scalar_timewarper",
     ###########################
     #########Encoding##########
     ###########################
     ###OneDConvEncoder###
     encoder_name="convolutional_encoder",
     emb_nonlinearity = "ReLU",
     emb_conv_layers_channels = [16,32,64,32],#[],#
     emb_conv_layers_strides = [1,2,2,2],#[],#
     emb_conv_layers_kernel_sizes = [3,3,3,3],#[],#
     emb_fc_layers_num_features = [],#[],#
     emb_dropout_probability = 0.0,
     ###########################
     #########Decoding##########
     ###########################
     ###OneDConvDecoder###
     decoder_name="convolutional_decoder_upsampling",
     ### Complicated enough model
     dec_gen_fc_layers_num_features = [25*NUM_CHANNELS*16],
     dec_gen_first_traj_len=25,
     dec_gen_conv_layers_channels = [NUM_CHANNELS*10,NUM_CHANNELS*10,NUM_CHANNELS],
     dec_gen_upsampling_factors = [2,2,2],
     dec_gen_conv_layers_kernel_sizes = [3,3,3],
     ###########################
     #####VectorTimewarping#####
     ###########################
     ###IdentityTimewarper### 
     vector_timewarper_name="identity_vector_timewarper"
     )

conv_no_dtw = dict(
     ###########################
     #####ScalarTimeWarping#####
     ###########################
     ###IdentityScalarTimewarper###
     scalar_timewarper_name = "identity_scalar_timewarper",
     ###########################
     #########Encoding##########
     ###########################
     ###OneDConvEncoder###
     encoder_name="convolutional_encoder",
     emb_conv_layers_channels = [16,16],#[],#
     emb_conv_layers_strides = [2,2],#[],#
     emb_conv_layers_kernel_sizes = [3,3],#[],#
     emb_fc_layers_num_features = [16],#[],#
     ###########################
     #########Decoding##########
     ###########################
     ###OneDConvDecoder###
     decoder_name="convolutional_decoder",
     ### Complicated enough model
     dec_gen_fc_layers_num_features = [100*NUM_CHANNELS,30*NUM_CHANNELS*10],
     dec_gen_first_traj_len=30,
     dec_gen_conv_layers_channels = [NUM_CHANNELS*10,NUM_CHANNELS*10,NUM_CHANNELS],
     dec_gen_conv_layers_strides = [2,2,2],
     dec_gen_conv_layers_kernel_sizes = [3,3,3],
     ###########################
     #####VectorTimewarping#####
     ###########################
     ###IdentityTimewarper### 
     vector_timewarper_name="identity_vector_timewarper"
     )

conv_dtw = dict(
     ###########################
     #####ScalarTimeWarping#####
     ###########################
     ###IdentityScalarTimewarper###
     scalar_timewarper_name = "identity_scalar_timewarper",
     ###########################
     #########Encoding##########
     ###########################
     ###OneDConvEncoder###
     encoder_name="convolutional_encoder",
     emb_conv_layers_channels = [16,16],#[],#
     emb_conv_layers_strides = [2,2],#[],#
     emb_conv_layers_kernel_sizes = [3,3],#[],#
     emb_fc_layers_num_features = [16],#[],#
     ###########################
     #########Decoding##########
     ###########################
     ###OneDConvDecoder###
     decoder_name="convolutional_decoder",
     ### Complicated enough model
     dec_gen_fc_layers_num_features = [100*NUM_CHANNELS,30*NUM_CHANNELS*10],
     dec_gen_first_traj_len=30,
     dec_gen_conv_layers_channels = [NUM_CHANNELS*10,NUM_CHANNELS*10,NUM_CHANNELS],
     dec_gen_conv_layers_strides = [2,2,2],
     dec_gen_conv_layers_kernel_sizes = [3,3,3],
     ###########################
     #####VectorTimewarping#####
     ###########################
     ###DTWTimewarper### 
     vector_timewarper_name="dtw_vector_timewarper",
     vector_timewarper_warps_recon_and_actual=True
     )

func_no_tw = dict(
     ###########################
     #####ScalarTimeWarping#####
     ###########################
     ###IdentityScalarTimewarper###
     scalar_timewarper_name = "identity_scalar_timewarper",
     ###########################
     #########Encoding##########
     ###########################
     ###OneDConvEncoder###
     encoder_name="convolutional_encoder",
     emb_conv_layers_channels = [16,16],#[],#
     emb_conv_layers_strides = [2,2],#[],#
     emb_conv_layers_kernel_sizes = [3,3],#[],#
     emb_fc_layers_num_features = [16],#[],#
     ###########################
     #########Decoding##########
     ###########################
     ###FunctionStyleDecoder###
     decoder_name="functional_decoder",
     dec_template_motion_hidden_layers=[500,500],
     ###########################
     #####VectorTimewarping#####
     ###########################
     ###IdentityTimewarper### 
     vector_timewarper_name="identity_vector_timewarper"
     )

func_tw = dict(
     ###########################
     #####ScalarTimeWarping#####
     ###########################
     ###ModeledParameterScalarTimewarper###
     scalar_timewarper_name = "modeled_scalar_timewarper",
     ## TimeWarpingRelated
     scaltw_granularity = 20,
     scaltw_emb_conv_layers_channels = [16,16],
     scaltw_emb_conv_layers_strides = [2,2],
     scaltw_emb_conv_layers_kernel_sizes = [3,3],
     scaltw_emb_fc_layers_num_features = [32],
     ###########################
     #########Encoding##########
     ###########################
     ###OneDConvEncoder###
     encoder_name="convolutional_encoder",
     emb_conv_layers_channels = [16,16],#[],#
     emb_conv_layers_strides = [2,2],#[],#
     emb_conv_layers_kernel_sizes = [3,3],#[],#
     emb_fc_layers_num_features = [16],#[],#
     ###########################
     #########Decoding##########
     ###########################
     ###FunctionStyleDecoder###
     decoder_name="functional_decoder",
     dec_template_motion_hidden_layers=[500,500],
     ###########################
     #####VectorTimewarping#####
     ###########################
     ###IdentityTimewarper### 
     vector_timewarper_name="identity_vector_timewarper"
     )

trans_func_tw = dict(
     ###########################
     #####ScalarTimeWarping#####
     ###########################
     ###ModeledParameterScalarTimewarper###
     scalar_timewarper_name = "modeled_scalar_timewarper",
     ## TimeWarpingRelated
     scaltw_granularity = 20,
     scaltw_emb_conv_layers_channels = [16,16],
     scaltw_emb_conv_layers_strides = [2,2],
     scaltw_emb_conv_layers_kernel_sizes = [3,3],
     scaltw_emb_fc_layers_num_features = [32],
     ###########################
     #########Encoding##########
     ###########################
     ###OneDConvEncoder###
     encoder_name="transformer_encoder",
     enc_attention_dims_per_head = 4,
     enc_append_time_dim = True,
     ###########################
     #########Decoding##########
     ###########################
     ###FunctionStyleDecoder###
     decoder_name="functional_decoder",
     dec_template_motion_hidden_layers=[500,500],
     ###########################
     #####VectorTimewarping#####
     ###########################
     ###IdentityTimewarper### 
     vector_timewarper_name="identity_vector_timewarper"
     )

func_side_no_tw = dict(
     ###########################
     #####ScalarTimeWarping#####
     ###########################
     ###IdentityScalarTimewarper###
     scalar_timewarper_name = "identity_scalar_timewarper",
     ###########################
     #########Encoding##########
     ###########################
     ###OneDConvEncoder###
     encoder_name="convolutional_encoder",
     emb_nonlinearity = "ReLU",
     emb_conv_layers_channels = [16,32,64,32],#[],#
     emb_conv_layers_strides = [1,2,2,2],#[],#
     emb_conv_layers_kernel_sizes = [3,3,3,3],#[],#
     emb_fc_layers_num_features = [],#[],#
     emb_dropout_probability = 0.0,
     ###########################
     #########Decoding##########
     ###########################
     ###FunctionStyleDecoder###
     decoder_name="functional_decoder_complicated",
     dec_template_motion_hidden_layers=[500,500],
     dec_complicated_function_hidden_dims = [16],
     dec_complicated_function_latent_size = 64,
     dec_template_custom_initialization_grad_t = 10.0,
     dec_template_use_custom_initialization = True,
     dec_template_custom_initialization_t_intercept_padding = 0.1,
     dec_complicated_only_side_latent = True,
     ###########################
     #####VectorTimewarping#####
     ###########################
     ###IdentityTimewarper### 
     vector_timewarper_name="identity_vector_timewarper",
     ####
     #### misc
     ####
     step_each_batch = True,
     learn_decoder_variance = False,
     dec_initial_log_noise_estimate = np.log(0.1**2).item(),
     pre_time_learning_epochs = 0,
     scalar_timewarping_lr = 0.0,
     scalar_timewarping_eps = 0.000001,
     scalar_timewarper_timereg = 0.0,
     scalar_timewarper_endpointreg = 0.0,
     scaltw_min_canonical_time = 0.0,
     scaltw_max_canonical_time = 1.0,
     dec_use_softplus=False,
     dec_use_elu=True,
     decoding_l2_weight_decay=0.0,
     decoding_spatial_derivative_regularization=0.0,
     dec_spatial_regularization_factor=1.0,
     decoding_lr = 0.0001,
     encoding_lr = 0.0001,
     decoding_eps = 0.0001,
     encoding_eps = 0.0001,
     useAdam = True,
     curv_loss_penalty_weight = 0.0,
     )

func_side_tw = dict(
     ###########################
     #####ScalarTimeWarping#####
     ###########################
     ###ModeledParameterScalarTimewarper###
     scalar_timewarper_name = "modeled_scalar_timewarper",
     ## TimeWarpingRelated
     scaltw_granularity = 50,
     scaltw_emb_conv_layers_channels = [16,32,32,64,64,64],
     scaltw_emb_conv_layers_strides = [1,2,1,2,1,2],
     scaltw_emb_conv_layers_kernel_sizes = [3,3,3,3,3,3],
     scaltw_emb_fc_layers_num_features = [],
     ###########################
     #########Encoding##########
     ###########################
     ###OneDConvEncoder###
     encoder_name="convolutional_encoder",
     emb_nonlinearity = "ReLU",
     emb_conv_layers_channels = [16,32,64,32],#[],#
     emb_conv_layers_strides = [1,2,2,2],#[],#
     emb_conv_layers_kernel_sizes = [3,3,3,3],#[],#
     emb_fc_layers_num_features = [],#[],#
     emb_dropout_probability = 0.0,
     ###########################
     #########Decoding##########
     ###########################
     ###FunctionStyleDecoder###
     decoder_name="functional_decoder_complicated",
     dec_template_motion_hidden_layers=[500,500],
     dec_complicated_function_hidden_dims = [16],
     dec_complicated_function_latent_size = 64,
     dec_template_custom_initialization_grad_t = 10.0,
     dec_template_use_custom_initialization = True,
     dec_template_custom_initialization_t_intercept_padding = 0.1,
     dec_complicated_only_side_latent = True,
     ###########################
     #####VectorTimewarping#####
     ###########################
     ###IdentityTimewarper### 
     vector_timewarper_name="identity_vector_timewarper",
     ####
     #### misc
     ####
     step_each_batch = True,
     learn_decoder_variance = False,
     dec_initial_log_noise_estimate = np.log(0.1**2).item(),
     pre_time_learning_epochs = 0,
     scalar_timewarping_lr = 0.0001,
     scalar_timewarping_eps = 0.000001,
     scalar_timewarper_timereg = 0.05,
     scalar_timewarper_endpointreg = 0,
     scaltw_min_canonical_time = 0.0,
     scaltw_max_canonical_time = 1.0,
     dec_use_softplus=False,
     dec_use_elu=True,
     decoding_l2_weight_decay=0.0,
     decoding_spatial_derivative_regularization=0.0,
     dec_spatial_regularization_factor=1.0,
     decoding_lr = 0.0001,
     encoding_lr = 0.0001,
     decoding_eps = 0.0001,
     encoding_eps = 0.0001,
     useAdam = True,
     curv_loss_penalty_weight = 0.0,
     )

self_attn_tw = dict(
     ###########################
     #####ScalarTimeWarping#####
     ###########################
     ###ModeledParameterScalarTimewarper###
     scalar_timewarper_name = "modeled_scalar_timewarper",
     ## TimeWarpingRelated
     scaltw_granularity = 20,
     scaltw_emb_conv_layers_channels = [16,16],
     scaltw_emb_conv_layers_strides = [2,2],
     scaltw_emb_conv_layers_kernel_sizes = [3,3],
     scaltw_emb_fc_layers_num_features = [32],
     ###########################
     #########Encoding##########
     ###########################
     ###OneDConvEncoder###
     #encoder_name="convolutional_encoder",
     #emb_conv_layers_channels = [32,64,32],#[],#
     #emb_conv_layers_strides = [2,2,1],#[],#
     #emb_conv_layers_kernel_sizes = [3,3,3],#[],#
     #emb_fc_layers_num_features = [128],#[],#
     ###SelfAttention###
     encoder_name="self_attention_transformer_encoder",
     emb_nonlinearity="ELU",
     enc_attention_dims_per_head = 128,
     enc_attention_num_heads = 16,
     enc_append_time_dim = True,
     ###########################
     #########Decoding##########
     ###########################
     ###FunctionStyleDecoder###
     decoder_name="functional_decoder_complicated",
     dec_template_motion_hidden_layers=[500,500],
     dec_complicated_function_hidden_dims = [16],
     dec_complicated_function_latent_size = 16,
     ###########################
     #####VectorTimewarping#####
     ###########################
     ###IdentityTimewarper### 
     vector_timewarper_name="identity_vector_timewarper"
     )

self_attn_no_tw = dict(
     ###########################
     #####ScalarTimeWarping#####
     ###########################
     ###ModeledParameterScalarTimewarper###
     scalar_timewarper_name = "identity_scalar_timewarper",
     ###########################
     #########Encoding##########
     ###########################
     ###OneDConvEncoder###
     #encoder_name="convolutional_encoder",
     #emb_conv_layers_channels = [32,64,32],#[],#
     #emb_conv_layers_strides = [2,2,1],#[],#
     #emb_conv_layers_kernel_sizes = [3,3,3],#[],#
     #emb_fc_layers_num_features = [128],#[],#
     ###SelfAttention###
     encoder_name="self_attention_transformer_encoder",
     emb_nonlinearity="ReLU",
     enc_attention_dims_per_head = 128,
     enc_attention_num_heads = 16,
     enc_append_time_dim = True,
     ###########################
     #########Decoding##########
     ###########################
     ###FunctionStyleDecoder###
     decoder_name="functional_decoder_complicated",
     dec_template_motion_hidden_layers=[500,500],
     dec_complicated_function_hidden_dims = [16],
     dec_complicated_function_latent_size = 16,
     ###########################
     #####VectorTimewarping#####
     ###########################
     ###IdentityTimewarper### 
     vector_timewarper_name="identity_vector_timewarper"
     )

self_attn_tw = dict(
     ###########################
     #####ScalarTimeWarping#####
     ###########################
     ###ModeledParameterScalarTimewarper###
     scalar_timewarper_name = "modeled_scalar_timewarper",
     ## TimeWarpingRelated
     scaltw_granularity = 20,
     scaltw_emb_conv_layers_channels = [16,16],
     scaltw_emb_conv_layers_strides = [2,2],
     scaltw_emb_conv_layers_kernel_sizes = [3,3],
     scaltw_emb_fc_layers_num_features = [32],
     ###########################
     #########Encoding##########
     ###########################
     ###OneDConvEncoder###
     #encoder_name="convolutional_encoder",
     #emb_conv_layers_channels = [32,64,32],#[],#
     #emb_conv_layers_strides = [2,2,1],#[],#
     #emb_conv_layers_kernel_sizes = [3,3,3],#[],#
     #emb_fc_layers_num_features = [128],#[],#
     ###SelfAttention###
     encoder_name="self_attention_transformer_encoder",
     emb_nonlinearity="ELU",
     enc_attention_dims_per_head = 128,
     enc_attention_num_heads = 16,
     enc_append_time_dim = True,
     ###########################
     #########Decoding##########
     ###########################
     ###FunctionStyleDecoder###
     decoder_name="functional_decoder_complicated",
     dec_template_motion_hidden_layers=[500,500],
     dec_complicated_function_hidden_dims = [16],
     dec_complicated_function_latent_size = 16,
     ###########################
     #####VectorTimewarping#####
     ###########################
     ###IdentityTimewarper### 
     vector_timewarper_name="identity_vector_timewarper"
     )
