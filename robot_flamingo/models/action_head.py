from typing import Optional, Tuple
from robot_flamingo.models.gentlyness import Gentlyness
import torch
import torch.nn as nn
from open_flamingo.src.helpers import PerceiverResampler
from robot_flamingo.models.normalizer import LinearNormalizer
from robot_flamingo.models.trajectory_gpt2 import get_gpt_model
# from .unets import *
import copy

def lstm_decoder(
    in_features: int, hidden_size: int, num_layers: int, policy_rnn_dropout_p: float
) -> torch.nn.Module:
    return nn.LSTM(
        input_size=in_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=False,
        batch_first=True,
        dropout=policy_rnn_dropout_p,
    )

class MLPTanhHead(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        """
        This class is outputs a value from -1, 1 and is used specifically for robot actions as these can be broken down into a range of [-1,1]
        """
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.mlp(x)

class MLPNohHead(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        """ 
        This class has no activation function meaning any real number is accepted. Perfect for embedded states
        """
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.mlp(x)

class MLPSigmoidHead(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        """
        This class will output between 0,1 perfect for gripper values which is never negative
        """
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x)

class MLPActionHead(torch.nn.Module):
    def __init__(self, hidden_size):
        """
        This class does the following:
        - Attaches to the hidden states of the RoboFlamingo model
        -splits robot actions into two parts robot movement and gripper movement
        num_head outputs 6 vectors which correspond to robot movement:
        x movement
        y movement
        z movement
        roll rotation
        pitch rotation
        yaw rotation 
        bin_head controlls the cripper in that it outputs 1 value which is either open or closed
        
        
        """
        super().__init__()
        self.hidden_size = hidden_size
        # Create a linear layer for each action
        self.num_head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )

        self.bin_head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = x[:, -1]  # pick up the last frame output
        x1 = self.num_head(x)
        x2 = self.bin_head(x).sigmoid()
        return x1, x2


class ActionDecoder(nn.Module):
    """
    This is just an interface of the different things the model needs to do like act, loss, loss_and_act and the forward pass
    """
    def act(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        robot_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def loss(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
        robot_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def loss_and_act(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
        robot_obs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _sample(self, *args, **kwargs):
        raise NotImplementedError

    def forward(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def clear_hidden_state(self) -> None:
        pass


class FCDecoder(ActionDecoder):
    """
    This class is the Fully connected decoder it takes the input from the VLM (openFlamingo) and turns its output into usefull information for the robot 
    Not used by the model
     
    """
    def __init__(
        
        self,
        in_features: int,
        window_size: int,
        history_len = None,
        out_features: int = 6,
        hidden_size: int = 1024,
        num_layers: int = 4,
        policy_rnn_dropout_p: float = 0.1,
        use_diff=False,
        last_action=False,
        fusion_mode='',
        use_state=False,
        return_feature=False,
        multi_step_action=1
        
    ):
        print("Using FC decoder")
        super(FCDecoder, self).__init__()
        self.return_feature = return_feature
        
        if use_state:
            #Uses current robot state information
            state_in_dim = 7
            state_out_dim = 128
            self.fc_state = MLPNohHead(state_in_dim, state_out_dim)
            in_features += state_out_dim
        
        if fusion_mode == 'two_way':
            in_features *= 2 #doubles the amount of features
        
        self.return_feature = return_feature
        self.in_features = in_features
        self.out_features = out_features
        #this uses previous model actions -> model is history aware
        self.window_size = window_size # -> this refers to the x amount of recent frames 
        self.multi_step_action = multi_step_action
        if history_len is None:
            history_len = window_size
        self.history_len = history_len
        self.history_memory = []

        self.use_diff = use_diff
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features//2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features//2, hidden_size),
        )
        if not use_diff: #if use_diff is false the system directly outputs two final heads ie 2 actions one for gripper the other for the robot arm.
            self.actions = MLPTanhHead(hidden_size, out_features) # predicts 6 continous actions
            self.gripper = MLPSigmoidHead(hidden_size, 1) #predicts 1 action for the gripper
        self.hidden_state = None
        self.hidden_size = hidden_size * history_len
        
        self.rnn_out = None
        self.last_action = last_action
        if self.use_diff:
            self.last_action = True
        # self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
        self.global_1d_pool = nn.AdaptiveMaxPool1d(1)
        #essentially all this work is done to turn a sequence of outputs from the vlm into a feature rich single vector summary

    def forward(  # type: ignore
            self,
            input_feature: torch.Tensor,
            h_0: Optional[torch.Tensor] = None,
            state_tensor = None,
    ):
        if self.return_feature:
            org_feat = copy.deepcopy(input_feature) 
            org_feat = org_feat.view(self.window_size, *org_feat.shape[1:])
            #This is relavent if we want to inspect the hidden features
        # reshape
        input_feature = self.mlp(input_feature) #Converts raw input features into the a hidden feature for action prediction
        input_feature = self.global_1d_pool(input_feature.permute(0, 2, 1)).squeeze(-1) #compress the tokens into one feature vector
        if self.use_diff: #unlike before when use_diff was false it will return features not actions 
            #Things like:
            #what objects are visible
            #where objects may be
            #what the instruction says
            #which visual tokens are relevant
            #what the model thinks the next action should depend on
            #temporal context from recent frames
            
            input_feature = input_feature.reshape(-1, self.window_size * input_feature.shape[1])
            return input_feature

        input_feature = input_feature.reshape(-1, self.window_size, input_feature.shape[1]) #action head looks over a sequence of of frames not individual ones
        #add robot state if available
        if state_tensor is not None:
            state_tensor = self.fc_state(state_tensor)
            state_tensor = state_tensor.reshape(-1, self.window_size, state_tensor.shape[-1])
            input_feature = torch.cat([input_feature, state_tensor], dim=-1)

        actions = self.actions(input_feature) #These are the actions of the robot
        gripper = self.gripper(input_feature) #actions of the gripper

        if self.return_feature: # If there is a original feature it will return this as well 
            return actions, gripper, org_feat
        else:
            return actions, gripper


class DeterministicDecoder(ActionDecoder):
    """
This is the decoder used by this checkpoint.

It takes hidden visual-language features from RoboFlamingo/OpenFlamingo,
optionally adds robot state information, processes the sequence with an LSTM,
and predicts:
- continuous robot arm actions
- gripper open/close action

Feature sequence → LSTM/RNN → action + gripper
"""
    def __init__(
        self,
        in_features: int,
        window_size: int,
        history_len = None,
        out_features: int = 6,
        hidden_size: int = 1024, #hidden lstm layers
        num_layers: int = 4, #lstm layers
        policy_rnn_dropout_p: float = 0.1,
        use_diff=False,
        last_action=False,
        fusion_mode='',
        use_state=False,
        multi_step_action=1,
        return_feature=False,
        pooling='max'
    ):
        print("using Deterministic decoder")
        super(DeterministicDecoder, self).__init__()
        self.fc_state = None
        self.use_state = use_state
        if use_state:
            #this aims to give the decoder information about the robots physical state
            print('Using state in decoder')
            state_in_dim = 7
            # state_out_dim = 256
            # in_features += state_out_dim
            # self.embed_arm_state = nn.Sequential(torch.nn.Linear(state_in_dim-1, state_out_dim), nn.ReLU())
            # self.embed_gripper_state = nn.Sequential(torch.nn.Embedding(2, state_out_dim), nn.ReLU()) # one-hot gripper state
            # self.embed_state = torch.nn.Linear(2*state_out_dim, state_out_dim)

            self.embed_arm_state = nn.Sequential(torch.nn.Linear(state_in_dim-1, in_features), nn.ReLU())
            self.embed_gripper_state = nn.Sequential(torch.nn.Embedding(2, in_features), nn.ReLU()) # one-hot gripper state
            self.embed_state = torch.nn.Linear(2*in_features, in_features) #This gives the current state representation of the robot arm: Essentially just states: This is what the arm currently is looking like
        
        if fusion_mode == 'two_way':
            in_features *= 2 #used for the camera on gripper and the 3rd person camera
        self.return_feature = return_feature #This then gets pooled so it can see both at the same time
        self.in_features = in_features
        self.out_features = out_features
        self.window_size = window_size
        self.multi_step_action = multi_step_action
        
        #this is setting up memory
        if history_len is None:
            history_len = window_size
        self.history_len = history_len
        self.history_memory = []
        self.rnn = lstm_decoder
        self.rnn = self.rnn(in_features, hidden_size, num_layers, policy_rnn_dropout_p) #sets up the LSTM decoder
        self.use_diff = use_diff
        self.fusion_mode = fusion_mode
        
        if not use_diff: #Again if the use_diff == False it creates 2 final predictions for the robot and gripper
            self.actions = MLPTanhHead(hidden_size, out_features*multi_step_action)
            self.gripper = MLPSigmoidHead(hidden_size, 1*multi_step_action)
        self.hidden_state = None
        self.hidden_size = hidden_size
       
        self.rnn_out = None
        #starting gentlyness values
        self.gentlyness = Gentlyness(log_activations=True)
        self.last_action = last_action
        
        
        if self.use_diff: #again it will return a feature representation instead of direct final predictions
            self.last_action = True
        if pooling == 'max':
            self.global_1d_pool = nn.AdaptiveMaxPool1d(1) #uses the most acitvated token signals
        else:
            self.global_1d_pool = nn.AdaptiveAvgPool1d(1) # uses a blend of all tokens
        
        if self.fusion_mode == 'two_way':
            if pooling == 'max':
                self.gripper_1d_max_pool = nn.AdaptiveMaxPool1d(1)
            else:
                self.gripper_1d_max_pool = nn.AdaptiveAvgPool1d(1)

    def clear_hidden_state(self) -> None:
        self.hidden_state = None
        self.history_memory = []

        if hasattr(self, "gentlyness"):
            self.gentlyness.reset(clear_logs=False)

    def forward(  # type: ignore
        self,
        input_feature: torch.Tensor,
        h_0: Optional[torch.Tensor] = None, #this is the initial lstm state
        state_tensor=None, # robot state
        return_feature=False
    ):
        
        
        # reshape
        if input_feature.dim() == 3:
            if self.fusion_mode == 'two_way': #if there are 2 cameras used
                input_feature = input_feature.reshape(-1, self.window_size, *input_feature.shape[1:])
                # This essentially looks at the camera feed from both the Main rgb camera and the gripper camera
                bs = int(input_feature.shape[0] // 2)
                
                rgb_feat = input_feature[:bs].view(bs*self.window_size, *input_feature.shape[2:])
                rgb_feat = self.global_1d_pool(rgb_feat.permute(0, 2, 1)).squeeze(-1) #pools only the robot/ main camera
                
                gripper_feat = input_feature[bs:].view(bs*self.window_size, *input_feature.shape[2:])
                gripper_feat = self.global_1d_pool(gripper_feat.permute(0, 2, 1)).squeeze(-1) # pools only the gripper camera
                
                input_feature = torch.cat([rgb_feat, gripper_feat], dim=-1)
            else: #if there is only one camera available
                input_feature = self.global_1d_pool(input_feature.permute(0, 2, 1)).squeeze(-1)
                
        input_feature = input_feature.reshape(-1, self.window_size, input_feature.shape[1]) #groups the input features into a feature sequence ready for lstm
        
        if self.return_feature: #saves a original copy of the original features
            org_feat = copy.deepcopy(input_feature) 
            org_feat = org_feat.view(self.window_size, org_feat.shape[-1])

        if state_tensor is not None and self.use_state: #this is used for the robot states if it is available 
            arm_state = state_tensor[..., :6] # b,len,state_dim-1
            arm_state_embeddings = self.embed_arm_state(arm_state) #takes first 6 state values and embedds them 7th dimension prob belongs to gripper therfore it must be excluded
            arm_state_embeddings = arm_state_embeddings.view(-1, self.window_size, arm_state_embeddings.shape[-1]) # b,len,h
            gripper_state = ((state_tensor[..., -1]+1.0) / 2).long() # b,len,1 converts gripper values to range from 0,1 
            gripper_state_embeddings = self.embed_gripper_state(gripper_state)
            gripper_state_embeddings = gripper_state_embeddings.view(-1, self.window_size, gripper_state_embeddings.shape[-1]) # b,len,h
            state_embeddings = torch.cat((arm_state_embeddings, gripper_state_embeddings), dim=2) # b,len,2h
            state_embeddings = self.embed_state(state_embeddings) # b,len,h

            # input_feature = torch.cat([input_feature, state_embeddings], dim=-1)
            input_feature = input_feature + state_embeddings #this is a representation of the physical situation of the robot itself
        #Lstm History stuff
        if not isinstance(self.rnn, nn.Sequential) and isinstance(self.rnn, nn.RNNBase):
            # print('history len:',self.history_len)
            if input_feature.shape[1] == 1:
                self.history_memory.append(input_feature)
                #does this only if history isnt full yet
                if len(self.history_memory) <= self.history_len:
                    # print('cur hist_mem len: {}'.format(len(self.history_memory)))
                    x, h_n = self.rnn(input_feature, self.hidden_state)
                    self.hidden_state = h_n
                    x = x[:, -1].unsqueeze(1)
                    self.rnn_out = x.squeeze(1)
                else:
                    # the hidden state need to be refreshed based on the history window 
                    #Only done if history gets too long
                    # print('hist_mem exceeded, refresh hidden state')
                    cur_len = len(self.history_memory)
                    for _ in range(cur_len - self.history_len):
                        self.history_memory.pop(0)
                    assert len(self.history_memory) == self.history_len
                    hist_feature = torch.cat(self.history_memory, dim=1)
                    self.hidden_state = None
                    x, h_n = self.rnn(hist_feature, self.hidden_state)
                    x = x[:, -1].unsqueeze(1)
                    self.rnn_out = x.squeeze(1)
            else:
                #this is done if there are multiple time steps at once -> it then passes this whole sequence into lstm
                # print('input feature lenght > 1', input_feature.shape)
                self.hidden_state = h_0
                x, h_n = self.rnn(input_feature, self.hidden_state)
                self.hidden_state = h_n
                if self.last_action: #this makes it sothat it takes the whole sequence as context but only predicts the action for the last frame
                    x = x[:, -1].unsqueeze(1)
                self.rnn_out = x.squeeze(1)
        else:
            raise NotImplementedError
        if self.use_diff: 
            return self.rnn_out #returns the output features of the lstm -> these are not robot actions but hidden representations
            #Contains things like:
            #current visual scene
            #language instruction
            #recent movement context
            #robot state, if enabled
            #history summarized by LSTM
        actions = self.actions(x) #final predictions based on history
        gripper = self.gripper(x)
        self.gentlyness.forward(actions, self.rnn_out) #This is our gentlyness stuff and the only thing i added/ changed in the policy head
        
        if self.return_feature:
            return actions, gripper, org_feat
        else:
            return actions, gripper

    def act(
        
        self,
        input_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Can be ignored as it isnt used"""
        pred_actions, self.hidden_state = self(
            input_feature, self.hidden_state
        )

        return pred_actions


class GPTDecoder(ActionDecoder):
    #Not used so no need to annotate
    def __init__(
        self,
        in_features: int,
        window_size: int,
        history_len = None,
        out_features: int = 6,
        hidden_size = None,
        num_layers: int = 4,
        policy_rnn_dropout_p: float = 0.1,
        last_action=False,
        use_diff=False,
        fusion_mode='',
        use_state=False,
        multi_step_action=1,
        return_feature=False,
        pooling='max',
        **kwargs
    ):  
        print("Using GPT decoder")
        super(GPTDecoder, self).__init__()
        
        if use_state:
            state_in_dim = 7
            state_out_dim = 128
            self.fc_state = MLPNohHead(state_in_dim, state_out_dim)
            in_features += state_out_dim
        
        if fusion_mode == 'two_way':
            in_features *= 2
        self.return_feature = return_feature
        self.in_features = in_features
        self.out_features = out_features
        self.window_size = window_size
        self.multi_step_action = multi_step_action
        if history_len is None:
            history_len = window_size
        self.history_len = history_len
        self.history_memory = []
        
        if hidden_size is None:
            hidden_size = in_features
        
        self.gpt = get_gpt_model(hidden_size, history_len)
        self.use_diff = use_diff
        self.fusion_mode = fusion_mode
        
        self.hidden_size = hidden_size
        if hidden_size != in_features:
            self.fc = nn.Linear(in_features, hidden_size)
        else:
            self.fc = nn.Identity()
        if not use_diff:
            self.actions = MLPTanhHead(hidden_size, out_features*multi_step_action)
            self.gripper = MLPSigmoidHead(hidden_size, 1*multi_step_action)
        self.hidden_state = None
        self.hidden_size = hidden_size
        self.rnn_out = None
        self.last_action = last_action
        if self.use_diff:
            self.last_action = True
        if pooling == 'max':
            self.global_1d_pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
        
        if self.fusion_mode == 'two_way':
            if pooling == 'max':
                self.gripper_1d_max_pool = nn.AdaptiveMaxPool1d(1)
            else:
                self.gripper_1d_max_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, input_feature: torch.Tensor):
        time_step=None
        attention_mask=None
        if input_feature.dim() == 3:
            input_feature = self.global_1d_pool(input_feature.permute(0, 2, 1)).squeeze(-1)
        input_feature = input_feature.reshape(-1, self.window_size, input_feature.shape[1]) # bs, seq_len, feat_dim
        input_feature = self.fc(input_feature)
        if input_feature.shape[1] == 1:
            self.history_memory.append(input_feature)
            
            if len(self.history_memory) <= self.history_len:
                hist_feature = torch.cat(self.history_memory, dim=1)
                x = self.gpt(hist_feature, time_step ,attention_mask)
                x = x[:, -1].unsqueeze(1)
                
            else:
                # the hidden state need to be refreshed based on the history window
                cur_len = len(self.history_memory)
                for _ in range(cur_len - self.history_len):
                    self.history_memory.pop(0)
                assert len(self.history_memory) == self.history_len
                hist_feature = torch.cat(self.history_memory, dim=1)
                x= self.gpt(hist_feature, time_step, attention_mask)
                x = x[:, -1].unsqueeze(1)
                
        else:
            x = self.gpt(input_feature, time_step, attention_mask)
            if self.last_action:
                x = x[:, -1].unsqueeze(1)
        actions = self.actions(x)
        gripper = self.gripper(x)
        return actions, gripper
    
    def get_pattern_name(self):
        return 'gpt_{}_'.format(self.hidden_size, )

class GPTDecoderActPad(ActionDecoder):
    #Doesnt use this one either so no need to annotate
    def __init__(
        self,
        in_features: int,
        window_size: int,
        use_vision = False,
        history_len = None,
        out_features: int = 6,
        hidden_size = None,
        last_action=False,
        use_diff=False,
        fusion_mode='',
        use_state=False,
        multi_step_action=1,
        return_feature=False,
        pooling='sampler',
        global_latent=10,
        **kwargs
    ):
        super(GPTDecoderActPad, self).__init__()
        
        if use_state:
            state_in_dim = 7
            state_out_dim = 128
            self.fc_state = MLPNohHead(state_in_dim, state_out_dim)
            in_features += state_out_dim
        
        if fusion_mode == 'two_way':
            in_features *= 2
        
        self.return_feature = return_feature
        self.in_features = in_features
        self.out_features = out_features
        self.window_size = window_size
        self.multi_step_action = multi_step_action
        if history_len is None:
            history_len = window_size
        self.history_len = history_len
        self.history_memory = []
        
        if hidden_size is None:
            hidden_size = in_features
        
        self.gpt = get_gpt_model(hidden_size, history_len, use_pe=False)
        self.use_diff = use_diff
        self.fusion_mode = fusion_mode
        
        self.hidden_size = hidden_size
        if hidden_size != in_features:
            self.fc = nn.Linear(in_features, hidden_size)
        else:
            self.fc = nn.Identity()
        if not use_diff:
            self.actions = MLPTanhHead(hidden_size, out_features*multi_step_action)
            self.gripper = MLPSigmoidHead(hidden_size, 1*multi_step_action)
        self.hidden_state = None
        self.hidden_size = hidden_size
        self.rnn_out = None
        self.last_action = last_action
        if self.use_diff:
            self.last_action = True
        self.global_latent = global_latent
        self.use_vision = use_vision
        if self.use_vision:
            self.vision_resampler = PerceiverResampler(dim=hidden_size)
        if pooling == 'sampler':
            self.global_1d_pool = PerceiverResampler(dim=hidden_size, depth=2, num_latents=global_latent)
        if pooling == 'max':
            self.global_1d_pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
        
        if self.fusion_mode == 'two_way':
            if pooling == 'max':
                self.gripper_1d_max_pool = nn.AdaptiveMaxPool1d(1)
            else:
                self.gripper_1d_max_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, input_feature: torch.Tensor, rgb=None):
        time_step=None
        attention_mask=None
        input_feature = self.global_1d_pool(input_feature.unsqueeze(1)).squeeze(1)
        input_feature = input_feature.view(-1, self.window_size, self.global_latent, input_feature.shape[-1]) # bs, seq_len, n_tok, feat_dim
        bs, seq_len, n_tok = input_feature.shape[:3]
        input_feature = self.fc(input_feature) # # bs, seq_len, n_tok, feat_dim
        attention_mask = torch.ones((bs, n_tok, seq_len), dtype=torch.long).to(input_feature.device)
        
        if input_feature.shape[1] == 1:
            self.history_memory.append(input_feature)
            
            if len(self.history_memory) <= self.history_len:
                hist_feature = torch.cat(self.history_memory, dim=1)
                x = self.gpt(hist_feature, time_step ,attention_mask)
                x = x[:, -1].unsqueeze(1)
                
            else:
                # the hidden state need to be refreshed based on the history window
                cur_len = len(self.history_memory)
                for _ in range(cur_len - self.history_len):
                    self.history_memory.pop(0)
                assert len(self.history_memory) == self.history_len
                hist_feature = torch.cat(self.history_memory, dim=1)
                x= self.gpt(hist_feature, time_step, attention_mask)
                x = x[:, -1].unsqueeze(1)
                
        else:
            x = self.gpt(input_feature, time_step, attention_mask)
            if self.last_action:
                x = x[:, -1].unsqueeze(1)
        actions = self.actions(x)
        gripper = nn.functional.sigmoid(self.gripper(x))
        return actions, gripper
    
    def get_pattern_name(self):
        return 'gpt_{}_'.format(self.hidden_size, )


class DiffusionDecoder(ActionDecoder):
    def __init__(
        self,
        feature_dim: int,
        window_size: int,
        history_len = None,
        horizon = 32,
        input_dim: int = 7, # dim of vectors to be diffused
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        n_timesteps=150,
        clip_denoised=False,
        predict_epsilon=True,
        normalizer = LinearNormalizer()
    ):
        super(DiffusionDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.horizon = horizon
        self.window_size = window_size
        if history_len is None:
            history_len = window_size
        self.history_len = history_len
        self.history_memory = []
        self.normalizer = normalizer
        self.data_dim = input_dim

        self.model = ConditionalUnet1D(
            input_dim,
            global_cond_dim=feature_dim,
            # global_cond_dim=None,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        # log calculation clipped because the posterior variance
        # is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )


    def get_loss_weights(self, action_weight, discount, weights_dict):
        """
        sets loss coefficients for trajectory
        action_weight   : float
            coefficient on first action loss
        discount   : float
            multiplies t^th timestep of trajectory loss by discount**t
        weights_dict    : dict
            { i: c } multiplies dimension i of observation loss by c
        """
        self.action_weight = action_weight

        dim_weights = torch.ones(self.action_dim, dtype=torch.float32)

        # set loss coefficients for dimensions of observation
        if weights_dict is None:
            weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        # decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)
        loss_weights = loss_weights.unsqueeze(1).clone()

        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, local_cond=None, global_cond=None, returns=None):

        if returns is not None: 
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, t, local_cond, global_cond, returns, use_dropout=False)
            epsilon_uncond = self.model(x, t, local_cond, global_cond, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w * (
                epsilon_cond - epsilon_uncond
            )
        else:
            epsilon = self.model(x, t, local_cond, global_cond)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, local_cond=None, global_cond=None, returns=None):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, local_cond=local_cond, global_cond=global_cond, returns=returns
        )
        noise = 0.5 * torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self, cond_data, cond_mask, local_cond=None, global_cond=None, returns=None, verbose=False, return_diffusion=False, **kwargs
    ):
        device = self.betas.device

        batch_size = cond_data.shape[0]
        x = torch.randn(
            size=cond_data.shape, 
            dtype=cond_data.dtype,
            device=cond_data.device
        )

        if return_diffusion:
            diffusion = [x]

        x[cond_mask] = cond_data[cond_mask]
        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):

            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            # 1. predict model output and replace sample
            x = self.p_sample(x, timesteps, local_cond, global_cond, returns)
            
            # 2. apply conditioning
            x[cond_mask] = cond_data[cond_mask]

            progress.update({"t": i})

            if return_diffusion:
                diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond_data, cond_mask, local_cond=None, global_cond=None, returns=None, action_seq_len=None, *args, **kwargs):
        """
        conditions : [ (time, state), ... ]
        """

        # horizon = action_seq_len or self.action_seq_len
        # batch_size = len(list(cond_data.values())[0])
        # shape = (batch_size, horizon, self.action_dim) # cond_data.shape
        return self.p_sample_loop(cond_data, cond_mask, local_cond, global_cond, returns, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample
    
    def forward(
        self,
        x,
        t,
        local_cond=None,
        global_cond=None,
        **kwargs
    ):
        return self.model(x, t, local_cond, global_cond)

    def act(
        self,
        input_feature: torch.Tensor,
    ) -> torch.Tensor:
        pred_actions, self.hidden_state = self(
            input_feature, self.hidden_state
        )

        raise NotImplementedError

if __name__ == "__main__":
    model = GPTDecoder(128, 24)
    in_feat = torch.randn((4*24, 12, 128))
    out = model(in_feat)
    print(out[0].shape, out[1].shape)
    pass