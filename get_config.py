import argparse


class BaseConfig:

    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.is_training = 1
        self.model_id = 'test'
        self.seq_len = 923
        self.label_len = 12
        self.pred_len = 58
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.dropout = 0.1
        self.use_norm = 1
        self.channel_independence = 1
        self.decomp_method = 'moving_avg'
        self.moving_avg = 24
        self.factor = 1


class PatchTSTConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.d_model = 128
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 128
        self.expand = 2
        self.d_conv = 4
        self.top_k = 5
        self.num_kernels = 6
        self.enc_in = 12
        self.dec_in = 7
        self.c_out = 7

        self.lr = 0.002
        self.batchsize = 256
        self.epochs = 40
        self.weightdecay = 0.5
        self.decaypatience = 4


class Dlinear(BaseConfig):
    def __init__(self):
        super().__init__()

        self.enc_in = 923
        self.moving_avg = 25

        self.lr = 0.002
        self.batchsize = 256
        self.epochs = 100
        self.weightdecay = 0.5
        self.decaypatience = 4


class TiDE(BaseConfig):
    def __init__(self):
        super().__init__()

        self.enc_in = 932

        self.lr = 0.002
        self.batchsize = 256
        self.epochs = 100
        self.weightdecay = 0.5
        self.decaypatience = 4


class Autoformer(BaseConfig):
    def __init__(self):
        super().__init__()

        self.enc_in = 923                      
        self.dec_in = 58           
        self.d_model = 58            
        self.batch_size = 64        
        self.embed = 'linear'            
        self.freq = 'h'        
        self.factor = 1             
        self.output_attention = False          
        self.moving_avg = 25               
        self.e_layers = 4         
        self.d_layers = 3         
        self.n_heads = 4            
        self.d_ff = 128          
        self.activation = 'gelu'        
        self.c_out = 58                   
        self.num_class = 10                       
        self.task_name = 'long_term_forecast'
        self.auxiliary = 15

                
        self.lr = 0.008       
        self.batchsize = 256           
        self.epochs = 100         
        self.weightdecay = 0.5          
        self.decaypatience = 3                     


class FITS(BaseConfig):
    def __init__(self):
        super().__init__()
        self.enc_in = 12
        self.individual = False

        self.lr = 0.001       
        self.batchsize = 256           
        self.epochs = 100         
        self.weightdecay = 0.9          
        self.decaypatience = 5                     


class FiLM(BaseConfig):
    def __init__(self):
        super().__init__()
        self.output_attention = 1
        self.e_layers = 3
        self.enc_in = 923
        self.ratio = 0.5

        self.lr = 0.002       
        self.batchsize = 128           
        self.epochs = 50         
        self.weightdecay = 0.5          
        self.decaypatience = 3                     


class NLinear(BaseConfig):
    def __init__(self):
        super().__init__()

        self.lr = 0.002
        self.batchsize = 256
        self.epochs = 100
        self.weightdecay = 0.5
        self.decaypatience = 4


class TimeXer(BaseConfig):
    def __init__(self):
        super().__init__()
        self.features = 'M'
        self.use_norm = True
        self.patch_len = 4
        self.enc_in = 923
        self.d_model = 128
        self.n_heads = 4
        self.e_layers = 3
        self.d_ff = 256
        self.factor = 5
        self.freq = 'h'        

        self.lr = 0.001
        self.batchsize = 256
        self.epochs = 50
        self.weightdecay = 0.5
        self.decaypatience = 3


class TimePerceiver(BaseConfig):
    def __init__(self):
        super().__init__()
        self.d_model = 128
        self.patch_len = 4
        self.enc_in = 923
        self.query_share = 1
        self.use_latent = 1
        self.num_latents = 8
        self.latent_dim = 128
        self.num_latent_blocks = 1
        self.n_heads = 2
        self.latent_d_ff = 128
        self.d_ff = 128
        self.seq_len = 12
        self.pred_len = 12


class iTransformer(BaseConfig):
    def __init__(self):
        super().__init__()
        self.output_attention = False          
        self.features = 'M'
        self.use_norm = True
        self.patch_len = 4
        self.enc_in = 12
        self.d_model = 128
        self.n_heads = 6
        self.e_layers = 4
        self.d_ff = 128
        self.factor = 5
        self.freq = 'h'        
        self.dec_in = 12
        self.d_layers = 1
        self.c_out = 12
                
        self.lr = 0.001       
        self.batchsize = 256           
        self.epochs = 100         
        self.weightdecay = 0.5          
        self.decaypatience = 5                     
        self.alpha = 0.1


class MambaSimple(BaseConfig):
    def __init__(self):
        super().__init__()
        self.features = 'M'
        self.use_norm = True
        self.patch_len = 24
        self.enc_in = 12
        self.d_model = 128
        self.n_heads = 4
        self.e_layers = 2
        self.d_ff = 128
        self.factor = 5
        self.freq = 'h'        
        self.expand = 2
        self.d_conv = 4
        self.c_out = 12

                
        self.lr = 0.001       
        self.batchsize = 32           
        self.epochs = 100         
        self.weightdecay = 0.5          
        self.decaypatience = 5                     


class Crossformer(BaseConfig):
    def __init__(self):
        super().__init__()
        self.features = 'M'
        self.enc_in = 12                      
        self.dec_in = 12           
        self.d_model = 128            
        self.batch_size = 256        
        self.embed = 'linear'            
        self.freq = 'h'        
        self.factor = 1             
        self.output_attention = False          
        self.moving_avg = 12               
        self.e_layers = 3         
        self.d_layers = 3         
        self.n_heads = 4            
        self.d_ff = 128          
        self.activation = 'gelu'        
        self.c_out = 1                   
        self.num_class = 10                       
        self.task_name = 'long_term_forecast'
        self.auxiliary = 15

                
        self.lr = 0.001       
        self.batchsize = 256           
        self.epochs = 30         
        self.weightdecay = 0.5          
        self.decaypatience = 1                     


class SparseTSF(BaseConfig):
    def __init__(self):
        super().__init__()

        self.enc_in = 12
        self.period_len = 24

                
        self.lr = 0.001       
        self.batchsize = 256           
        self.epochs = 100         
        self.weightdecay = 0.8          
        self.decaypatience = 3                     


class xlstm(BaseConfig):
    def __init__(self):
        super().__init__()
        self.context_points = 168
        self.target_points = 24
        self.enc_in = 12
        self.n2 = 256
        self.embedding_dim = 256

        self.lr = 0.001       
        self.batchsize = 256           
        self.epochs = 100         
        self.weightdecay = 0.8          
        self.decaypatience = 3                     


def get_config(model_name: str):
    if model_name == 'PatchTST':     
        return PatchTSTConfig()
    elif model_name == 'DLinear':     
        return Dlinear()
    elif model_name == 'TiDE':     
        return TiDE()
    elif model_name == 'Autoformer':     
        return Autoformer()
    elif model_name == 'FITS':     
        return FITS()
    elif model_name == 'FiLM':     
        return FiLM()
    elif model_name == 'NLinear':     
        return NLinear()
    elif model_name == 'TimeXer':     
        return TimeXer()
    elif model_name == 'iTransformer':     
        return iTransformer()
    elif model_name == 'MambaSimple':     
        return MambaSimple()
    elif model_name == 'Crossformer':     
        return Crossformer()
    elif model_name == 'SparseTSF':     
        return SparseTSF()
    elif model_name == 'xlstm':
        return xlstm()
    elif model_name=='TimePerceiver':
        return TimePerceiver()


    else:
        raise ValueError(f"Unsupported model name: {model_name}")
