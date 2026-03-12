####### Trend and seasonality decompostion using Wavelet and forecasting using a hybrid model of LSTM and mlp-mixer


import torch
from torch import nn
from torch.nn import functional as F
import math



class MLPMixer(nn.Module):
    def __init__(self, patch_size, num_patches, patch_hidden_dim, time_hidden_dim):
        super(MLPMixer, self).__init__()

        self.patch_size = patch_size
        self.num_patches = num_patches

        self.patch_mix = nn.Sequential(
            nn.LayerNorm(self.num_patches),
            nn.Linear(self.num_patches, patch_hidden_dim),
            nn.GELU(),
            nn.Linear(patch_hidden_dim, self.num_patches)
        )
        
        self.time_mix = nn.Sequential(
            nn.LayerNorm(self.patch_size),
            nn.Linear(self.patch_size, time_hidden_dim),
            nn.GELU(),
            nn.Linear(time_hidden_dim, self.patch_size)
        )
      
    def forward(self, x):
        batch_size, context_length = x.shape
        x = x.view(batch_size, self.num_patches, self.patch_size)

        
        x = x + self.patch_mix(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.time_mix(x)
        
        x = x.reshape(batch_size, self.num_patches * self.patch_size)
        return x
    


        # other tasks not implemented


class RNNBlock(nn.Module):
    #def __init__(self, thetas_dim, backcast_length, forecast_length, embed_dim=8, input_size=8, hidden_size=8, rnn_type="LSTM"):
    def __init__(self, thetas_dim, backcast_length, embed_dim=8, input_size=8, hidden_size=8, rnn_type="LSTM"):
        super(RNNBlock, self).__init__()
        self.backcast_length = backcast_length
        #self.forecast_length = forecast_length
        self.embed_dim= embed_dim
        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, batch_first=True)

        self.theta_b_fc = nn.Linear(backcast_length, thetas_dim, bias=False)
        # self.theta_f_fc = nn.Linear(backcast_length, thetas_dim, bias=False)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        # self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        batch_size, context_length = x.shape
        seq_length = context_length // self.embed_dim
        x = x.view(batch_size, seq_length, self.embed_dim)
     
        output, (hn, _) = self.rnn(x)  # hn shape: (1, batch_size, hidden_size)
        output = output.reshape(batch_size, seq_length * self.embed_dim)
        theta_b = self.theta_b_fc(output)
        # theta_f = self.theta_f_fc(output)

        backcast = self.backcast_fc(theta_b)
        # forecast = self.forecast_fc(theta_f)

        return backcast#, forecast





class MLPMixerBlock(nn.Module):
    # def __init__(self, hidden_dim, thetas_dim, device, backcast_length=10, forecast_length=5, patch_size=8, num_patches=21):
    def __init__(self, hiddend_dim, thetas_dim, device, backcast_length=10, patch_size=8, num_patches=21):
        super(MLPMixerBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        # self.forecast_length = forecast_length
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.device = device
        
        self.mlpMixer = MLPMixer(self.patch_size, self.num_patches, self.hidden_dim, self.hidden_dim)
        
        self.theta_b_fc = nn.Linear(backcast_length, thetas_dim, bias=False)
        # self.theta_f_fc = nn.Linear(backcast_length, thetas_dim, bias=False)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        # self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):

        x = self.mlpMixer(x)


        theta_b = self.theta_b_fc(x)
        #theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)
        # forecast = self.forecast_fc(theta_f)

        return backcast#, forecast
    



class Model(nn.Module):
    def __init__(
            self,
            device=torch.device('cpu'),
            num_blocks_per_stack=3,
            # forecast_length=24,
            backcast_length=168,
            patch_size=8,
            num_patches=21,
            thetas_dim=8,
            embed_dim=8,
            hidden_dim=256,
            num_heads=2,
            ff_hidden_dim=256,
            context_length= 168,
            dropout= 0.1,
            #share_weights_in_stack=False,
            #positional_embed_flag=True,
            num_classes=1
    ):
        super(Model, self).__init__()

        self.device = device
        # self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_blocks_per_stack = num_blocks_per_stack
        # self.share_weights_in_stack = share_weights_in_stack
        self.thetas_dim = thetas_dim
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.num_classes = num_classes

        # self.log_sigma_trend = nn.Parameter(torch.tensor(0.0))
        # self.log_sigma_season = nn.Parameter(torch.tensor(0.0))

        ########Embeddings####
        self.proj_linear_trend = nn.Linear(context_length, backcast_length)
        self.proj_linear_season = nn.Linear(context_length, backcast_length)
        self.dropout = nn.Dropout(p=dropout)

        # Trend stack (RNN(Type-LSTM)-based)
      
        #########_init__(self, thetas_dim, backcast_length, forecast_length, embed_dim=8, input_size=8, hidden_size=8, rnn_type="LSTM"):
        self.trend_stack = nn.ModuleList([
            RNNBlock(thetas_dim=self.thetas_dim,
                    backcast_length=self.backcast_length,
                    # forecast_length=self.forecast_length,
                    embed_dim= self.embed_dim,
                    input_size=self.embed_dim,
                    hidden_size=self.embed_dim,             
                    rnn_type="LSTM")
            for _ in range(self.num_blocks_per_stack)
        ])

        # Seasonality stack (MLP-Mixer-based)
        self.seasonality_stack = nn.ModuleList([
            MLPMixerBlock(hidden_dim=self.hidden_dim, 
                          thetas_dim=self.thetas_dim,
                          device=self.device,
                          backcast_length=self.backcast_length,
                          # forecast_length=self.forecast_length,
                          patch_size=self.patch_size,
                          num_patches=self.num_patches)
            for _ in range(self.num_blocks_per_stack)
        ])

        # Classification head
        self.classifier = nn.Sequential(
                nn.Linear(backcast_length * 2, ff_hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(ff_hidden_dim, num_classes)
        )

        self.to(self.device)

    def forward(self, trend_input, seasonality_input):
        """
        trend_input: [batch_size, backcast_length]
        seasonality_input: [batch_size, backcast_length]
        returns: logits [batch_size, num_classes]
        """
        # trend_forecast = torch.zeros(trend_input.size(0), self.forecast_length).to(self.device)
        # seasonality_forecast = torch.zeros(seasonality_input.size(0), self.forecast_length).to(self.device)

        trend_input = self.dropout(self.proj_linear_trend(trend_input))
        seasonality_input = self.dropout(self.proj_linear_season(seasonality_input))

        # Trend path (LSTM-based)
        for block in self.trend_stack:
            #trend_backcast, trend_forecast_block = block(trend_input)
            trend_backcast = block(trend_input)
            trend_input = trend_input - trend_backcast
            # trend_forecast += trend_forecast_block

        # Seasonality path (MLP-Mixer-based)
        for block in self.seasonality_stack:
            #season_backcast, season_forecast_block = block(seasonality_input)
            season_backcast = block(seasonality_input)
            seasonality_input = seasonality_input - season_backcast
            # seasonality_forecast += season_forecast_block

        # Concatenate residual representations and classify
        combined = torch.cat([trend_input, seasonality_input], dim=-1)
        logits = self.classifier(combined)

        return logits

        #return trend_forecast, seasonality_forecast


    


