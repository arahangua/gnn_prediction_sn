#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:01:31 2021

@author: th
"""

import numpy as np
import neo
import quantities as pq
import elephant

from itertools import combinations 
from elephant.statistics import time_histogram
from elephant.conversion import BinnedSpikeTrain
from viziphant.rasterplot import rasterplot
import parmap
from elephant.spike_train_correlation import spike_time_tiling_coefficient

import bct


# for wp1 

class Spiketrain:
    def __init__(self, spktimes, spktemps):
        self.spktimes = spktimes
        self.spktemps = spktemps
        
    def set_trains(self):
        uniq_ids=np.unique(self.spktemps)
      
        spktime_list=[]
        for ids in uniq_ids:
            loc=np.where(self.spktemps==ids)[0]
            spktime_list.append(self.spktimes[loc])
        
        self.spktime_list = spktime_list
        self.uniq_ids = uniq_ids.astype(int)
        
        return spktime_list, uniq_ids
    
        
            
        
    def to_neotrain(self):
        st_time_ms = np.floor(np.min(self.spktimes))
        ed_time_ms= np.ceil(np.max(self.spktimes))+1
        
        
        spk_mean_rate=[]
        for spktrain in self.spktime_list:
            if(len(spktrain)==1):
                neo_spk_train = neo.SpikeTrain(spktrain*pq.ms, units=pq.ms, t_start=st_time_ms, t_stop=ed_time_ms)
            else:
                neo_spk_train = neo.SpikeTrain(np.squeeze(spktrain)*pq.ms, units=pq.ms, t_start=st_time_ms, t_stop=ed_time_ms)
            spk_mean_rate.append(neo_spk_train)
            
        self.neotrain = spk_mean_rate    
            
        return spk_mean_rate
    
    def to_countmat(self, binsize):
        st_time_ms = np.floor(np.min(self.spktimes))
        ed_time_ms= np.ceil(np.max(self.spktimes))+1
        
        
        spk_mean_rate=[]
        for spktrain in self.spktime_list:
            neo_spk_train = neo.SpikeTrain(np.squeeze(spktrain)*pq.ms, units=pq.ms, t_start=st_time_ms, t_stop=ed_time_ms)
            histogram_count = time_histogram([neo_spk_train], binsize*pq.ms, output='counts')
            spk_mean_rate.append(np.squeeze(np.asarray(histogram_count)))
           
        spk_mean_rate=np.asarray(spk_mean_rate)
        
        self.countmat = spk_mean_rate
    
        return spk_mean_rate
    
        
    def to_ratemat(self, binsize):
        st_time_ms = np.floor(np.min(self.spktimes))
        ed_time_ms= np.ceil(np.max(self.spktimes))+1
        
        
        spk_mean_rate=[]
        for spktrain in self.spktime_list:
            neo_spk_train = neo.SpikeTrain(np.squeeze(spktrain)*pq.ms, units=pq.ms, t_start=st_time_ms, t_stop=ed_time_ms)
            histogram_count = time_histogram([neo_spk_train], binsize*pq.ms, output='rate')
            spk_mean_rate.append(np.squeeze(np.asarray(histogram_count)))
           
        spk_mean_rate=np.asarray(spk_mean_rate)
        
        self.countmat = spk_mean_rate
    
        return spk_mean_rate
    
    
    
    def to_binned(self, binsize):
        st_time_ms = np.floor(np.min(self.spktimes))
        ed_time_ms= np.ceil(np.max(self.spktimes))+1
        
        spk_vec=[]
        for spktrain in self.spktime_list:
            neo_spk_train = neo.SpikeTrain(np.squeeze(spktrain)*pq.ms, units=pq.ms, t_start=st_time_ms, t_stop=ed_time_ms)
            spk_vec.append(neo_spk_train)
            
        
        bst = BinnedSpikeTrain(spk_vec, bin_size=binsize * pq.ms)
        self.binned_mat = bst
        
        return bst
        

    
    def compute_corr(self, pca_binsize):
        # compute pearson correlation for all data
        
        corr_mat = np.corrcoef(self.to_countmat(pca_binsize))
        self.corr_mat = corr_mat
        
        return corr_mat
    
        
        
        

    
            
    
    def plot_raster(self, dot_size):
        neotrain = self.to_neotrain()
        
        rasterplot(neotrain, s=dot_size, c='black')
        
    def parallel_fncch(self, entry, neotrains, binsize):
        #adapting pastore et al. 2018
        #binsize =1
        #n_cell = len(neotrains)
        # weight_mat= np.zeros((n_cell,n_cell))
        # delay_mat = np.zeros((n_cell, n_cell))
        
        idx1 = entry[0]
        idx2 = entry[1]
        
        binned1 = BinnedSpikeTrain(neotrains[idx1], bin_size=binsize * pq.ms)
        binned2 =  BinnedSpikeTrain(neotrains[idx2], bin_size=binsize * pq.ms)
        cch, lags= elephant.spike_train_correlation.cross_correlation_histogram(binned1, binned2, window=[-50,50])
        
        #looking into both directions
        
        cch = np.squeeze(cch)
        
        # normalized cch 
        total_spikes = np.sqrt(np.sum(binned1.to_array()) *np.sum(binned2.to_array()))
        
        cch = cch/total_spikes
        
        forward_win = cch[51:76] # 25ms 
        backward_win = cch[25:50] # 25ms 
        #flip it
        backward_win = backward_win[::-1]
        
        
        norm_forward = forward_win - np.mean(forward_win)
        norm_backward = backward_win - np.mean(backward_win)
        
        prepost = [idx1, idx2, np.max(abs(norm_forward)), np.argmax(abs(norm_forward)), norm_forward]
        postpre = [idx2, idx1, np.max(abs(norm_backward)), np.argmax(abs(norm_backward)), norm_backward]
        
        result = [prepost, postpre]
        return result
    
        
    def make_conmat(self, result):
        n_neurons = len(self.spktime_list)
        weight_mat = np.zeros((n_neurons, n_neurons))
        lag_mat = np.zeros((n_neurons, n_neurons))
        ex_in_mat = np.zeros((n_neurons, n_neurons))
        
        for entry in result:
            prepost = entry[0]
            postpre = entry[1]
            
            weight_mat[prepost[0], prepost[1]] = prepost[2]
            if(prepost[4][prepost[3]]==np.min(prepost[4])):
                ex_in_mat[prepost[0], prepost[1]]= -1
            elif(prepost[4][prepost[3]]==np.max(prepost[4])):
                ex_in_mat[prepost[0], prepost[1]]= 1
                
            
            weight_mat[postpre[0], postpre[1]] = postpre[2]
            if(postpre[4][postpre[3]]==np.min(postpre[4])):
                ex_in_mat[postpre[0], postpre[1]]= -1
            elif(postpre[4][postpre[3]]==np.max(postpre[4])):
                ex_in_mat[postpre[0], postpre[1]]= 1
            
            
            
            lag_mat[prepost[0], prepost[1]] = prepost[3]
            lag_mat[postpre[0], postpre[1]] = postpre[3]
            
        return weight_mat, lag_mat, ex_in_mat
            
        
    def make_distance_vec(self, probe_idx, xy_pos):
        distance_vec=[]
        pre = probe_idx[0]
        post = probe_idx[1]
        
        for ii in range(len(pre)):
            
            pos1 = xy_pos[pre[ii],:]
            pos2 = xy_pos[post[ii],:]
            distance_vec.append(np.linalg.norm(pos1 - pos2))
            
        return np.array(distance_vec)

    def compute_con_density(self, dale_mat):
        tot_con= dale_mat.shape[0]**2 - dale_mat.shape[0]
        exist_con = np.sum(dale_mat!=0)
        density = exist_con/tot_con
        return density
    
    def apply_dale_law(self, ex_in_mat_after_prop):
        dale_mat = np.copy(ex_in_mat_after_prop)
        for row in range(ex_in_mat_after_prop.shape[0]):
           count_ex = np.sum(ex_in_mat_after_prop[row,:]==1)
           count_in = np.sum(ex_in_mat_after_prop[row,:]==-1)
           
           if(count_ex>=count_in):
               dale_mat[row,np.where(ex_in_mat_after_prop[row,:]==-1)[0]]=0
           else:
               dale_mat[row,np.where(ex_in_mat_after_prop[row,:]==1)[0]]=0
        return dale_mat
    
    def compute_fncch_connectivity(self, xy_pos, sensitivity, binsize=1, n_cpu=16):
                
        n_neurons = len(self.spktime_list)
        for_idx = np.arange(n_neurons)
        neotrains = self.to_neotrain()
        
        combi = combinations(for_idx, 2)
        combi_list=list(combi)
        
        
        result = parmap.map(self.parallel_fncch, combi_list, neotrains, binsize, pm_processes=n_cpu, pm_pbar=True)
   
        weight_mat, lag_mat, ex_in_mat = self.make_conmat(result)
        
        excitatory_idx = np.where(ex_in_mat>0)
        inhibitory_idx = np.where(ex_in_mat<0)
        
        thres_weight_mat = np.copy(weight_mat)
        
         # spatio temporal filtering 
        prop_velocity = 600 # 600um/ms
        # xy_pos = self.get_xy_positions(sorted_time)
               
        probe_idx = np.where(thres_weight_mat>0)
        lags_to_probe = lag_mat[probe_idx]
        
        distance_vec = self.make_distance_vec(probe_idx, xy_pos)
        lag_ceil = distance_vec / prop_velocity
        
        idx_to_kick = []
        for ii, lag_ in enumerate(lags_to_probe):
            if(lag_ < lag_ceil[ii]):
                idx_to_kick.append(ii)
                
        # reflect this result
        pre = probe_idx[0]
        post = probe_idx[1]
        ex_in_mat_raw = np.copy(ex_in_mat)
        con_result=dict()
        con_result['ex_in_mat_raw']=ex_in_mat_raw
               
        thres_weight_mat[pre[idx_to_kick], post[idx_to_kick]] = 0
        #ex_in_mat_before_prop = np.copy(ex_in_mat)
        ex_in_mat[pre[idx_to_kick], post[idx_to_kick]] = 0
        
        excitatory_idx = np.where(ex_in_mat>0)
        inhibitory_idx = np.where(ex_in_mat<0)
        
        #con_result['ex_in_mat_before_prop']=ex_in_mat_before_prop
        con_result['raw_after_prop']=np.copy(thres_weight_mat)
        con_result['weight_raw']=np.copy(weight_mat)
        con_result['ex_in_mat_after_prop']=np.copy(ex_in_mat)
        #apply pastore et al. 2018 hard threshold mu + sigma for inhibitory, mu + 2*sigma for excitatory
        weight_mat = thres_weight_mat
        
        np.fill_diagonal(weight_mat, np.nan)
        
        mu_ex = np.nanmean(abs(weight_mat[excitatory_idx]))
        sigma_ex = np.nanmean(abs(weight_mat[excitatory_idx]))

        mu_in = np.nanmean(abs(weight_mat[inhibitory_idx]))
        sigma_in = np.nanmean(abs(weight_mat[inhibitory_idx]))
        
        #sns.heatmap(weight_mat)
  
        to_zero_ex=np.where(weight_mat[excitatory_idx]< (mu_ex +2*sigma_ex*sensitivity))
        to_zero_in=np.where(weight_mat[inhibitory_idx]< (mu_in +sigma_in*sensitivity))
        
        thres_weight_mat[excitatory_idx[0][to_zero_ex], excitatory_idx[1][to_zero_ex]] = 0
        thres_weight_mat[inhibitory_idx[0][to_zero_in], inhibitory_idx[1][to_zero_in]] = 0
        
        ex_in_mat[excitatory_idx[0][to_zero_ex], excitatory_idx[1][to_zero_ex]] = 0
        ex_in_mat[inhibitory_idx[0][to_zero_in], inhibitory_idx[1][to_zero_in]] = 0
        
        con_result['final_weight_mat']= thres_weight_mat
        con_result['lag_mat'] = lag_mat
        con_result['ex_in_mat'] = ex_in_mat
        con_result['sensitivity']=sensitivity
        
        # further filtering e.g. dale's law 
        
        # application of dale's law
        
        ex_in_mat_after_prop = np.copy(con_result['ex_in_mat_after_prop'])
        
        
        # exists ihibitory? 
        dale_mat = self.apply_dale_law(ex_in_mat_after_prop)        
        # check connection density. not used now.
        dale_density = self.compute_con_density(dale_mat)
               
        # dale treatment of vanilla result 
        van_graph = self.apply_dale_law(con_result['ex_in_mat'])        
        van_weight = np.copy(con_result['final_weight_mat'])
        van_weight[van_graph==0]=0
        
        con_result['dale_mat']= van_graph
        con_result['dale_mat_raw']=dale_mat
        con_result['dale_weight']=van_weight
        
        #check number of components for vanilla thresholded matrix
        #undirected treatment 
        
       
        van_graph = (van_graph + van_graph.T) /2
        van_graph[van_graph!=0]=1
        comps, comps_size=bct.algorithms.get_components(van_graph)
        
        print('num of connected components :' + str(np.sum(comps_size>=2)))
        print('density of generated graph (converted to an undirected graph) :' + str(self.compute_con_density(van_graph)))
        print('used sensitivity : ' +str(sensitivity))
        # apply thesholding 
        
        con_result['con_density']=self.compute_con_density(van_graph)
        con_result['comps_viable']=np.sum(comps_size>=2)
        con_result['comp_sizes']=comps_size
        
           
        
        return con_result
    

    
    def standardize_array(self, countmat):
        #adapted from Abid et al. 2018
        standardized_array =  (countmat-np.mean(countmat,axis=0)) / np.std(countmat,axis=0)
        return np.nan_to_num(standardized_array)

        
            
    def make_pos_dict(self, xy_pos):
        pos_dict=dict()
        for ii in range(xy_pos.shape[0]):
            pos_dict[ii] = xy_pos[ii,:]
            
        return pos_dict
            

    def get_nx_edge_weights(self, weights):
        wei_vec =[]
        for entry in weights:
            wei_vec.append(entry[-1])
            
        return np.array(wei_vec)

    def flatten_list_1d(self, act_ratio):
        ph = np.empty((1,0))
        ph = np.squeeze(ph)
        
        for entry in act_ratio:
            ph = np.concatenate((ph, entry))
            
        return ph
    
    def node_mask_gen(self, group_list, label_vec):
       
        n_neurons = len(self.spktime_list)
        
        place_idx = np.arange(n_neurons)
        place_hol = np.zeros(place_idx.shape)
        
        place_hol[:]=label_vec[-1]
        
        for ii, entry in enumerate(group_list):
            bool_idx=np.isin(place_idx, entry)
            place_hol[bool_idx]= label_vec[ii]
            
        #check 
        
        return place_hol

               
    def load_fncch_connectivity(self, sorted_time):
        
        con_result = np.load(sorted_time + '/con_fncch_result.npy', allow_pickle=True).item()
        
        thres_weight_mat= con_result['final_weight_mat']
        lag_mat = con_result['lag_mat'] 
        ex_in_mat = con_result['ex_in_mat'] 
        
        return thres_weight_mat, lag_mat, ex_in_mat
    
    


        
  
        
        
        
        

     
           

    def shuffle_trains(self, shuffle_iter):
        
        
        neotrain = self.to_neotrain()
        
        shuf_append=[]
        for _ in range(shuffle_iter):
            shuffled_train=[]
            for neo_entry in neotrain:
                random_surrogate = elephant.spike_train_surrogates.shuffle_isis(neo_entry) #default shuffling is isis
                shuffled_train.append(random_surrogate[0])
            shuf_append.append(shuffled_train)
        
                    
        
        # savefol = neural_assembly_utils.return_savepath(file, config)
        # np.save(savefol + '/shuf_trains_' + alias, shuf_append)
        return shuf_append
       
    def get_min_corr(self, results):
        min_corr_vec = []
        
        for entry in results:
            min_corr_vec.append(np.min(entry))
            
        return np.array(min_corr_vec)

    def get_corr_mat(self, shuf_trains, pca_bin=20, n_cpu=16):
        
    
        neotrain = self.to_neotrain()
        spk_mean_rate=[]
        for neo_entry in neotrain:
            histogram_count = time_histogram([neo_entry], pca_bin*pq.ms, output='counts')
            spk_mean_rate.append(np.squeeze(np.asarray(histogram_count)))
        spk_mean_rate = np.array(spk_mean_rate)
        
        corr_mat = np.corrcoef(spk_mean_rate)
        #take care of nans
        corr_mat[np.isnan(corr_mat)]=0
        
        corr_mat_result = dict()
        corr_mat_result['corr_mat_raw']=corr_mat
        
            
        
        results = parmap.map(self.parallel_corr, shuf_trains, pca_bin, pm_processes=n_cpu, pm_pbar = False)
        
                
        eff_corr = self.nontrivial_corr(corr_mat, results) #get those survive the test
        shuf_min_corr = self.get_min_corr(results)        
        
                   
        corr_mat_result['corr_mat_eff']=eff_corr
        corr_mat_result['shuffled_neg']=shuf_min_corr
        
        print('done computing PCC FC mat')
                
        return  corr_mat_result
    
    def nontrivial_corr(self, corr_mat, results):
        
        
        tp_corr = np.copy(corr_mat)
        np.fill_diagonal(tp_corr,0)
        
        tp_corr = np.round(tp_corr,8) #use double
        
        surr_collect=[]
        for entry in results:
            tp_surr = entry
            np.fill_diagonal(tp_surr,0)
            tp_surr = np.round(tp_surr,8) #use double
            surr_collect.append(tp_surr)
        
        surr_collect= np.array(surr_collect)
        
        #get p-value like metric 
        
        sorted_surr = np.sort(surr_collect,axis=0)
        
        #first check any values being lower than surr
        diff_corr = tp_corr - sorted_surr[-1,:,:]
        
        #look for negative values 
        
        neg_tuple= np.where(diff_corr<0)
        #neg_vals= diff_corr[neg_tuple]
        
        neg_arr = np.array(neg_tuple)
        
        pval_col =[]
        for col in range(neg_arr.shape[1]):
            loc=np.searchsorted(sorted_surr[:,neg_arr[0,col], neg_arr[1,col] ], tp_corr[neg_arr[0,col],neg_arr[1,col]])
            pval = (sorted_surr.shape[0]-loc)/sorted_surr.shape[0]
            pval_col.append(pval)
        
        pval_col= np.array(pval_col)
        # p val matrix 
        
        pval_mat = np.zeros((tp_corr.shape))
        pval_mat[neg_tuple]= pval_col
        
        #np.sum(pval_col<0.05)
        
        # return corr_mats
        masked_corr_mat = np.copy(tp_corr)
        masked_corr_mat[np.where(pval_mat>0)] = 0
        masked_eff_corrmat = np.copy(diff_corr)
        masked_eff_corrmat[np.where(pval_mat>0)] = 0
        
        tp_result=dict()
        tp_result['p_val_mat']= pval_mat
        tp_result['masked_corr']=masked_corr_mat
        tp_result['masked_eff']=masked_eff_corrmat
        tp_result['raw_corr_mat'] = tp_corr
            
        
        return tp_result

            


    def shuffle_preserve_isi_corr_control(self, shuffled_train, pca_bin):

        
        spk_mean_rate=[]
        for neo_entry in shuffled_train:
            histogram_count = time_histogram([neo_entry], pca_bin*pq.ms, output='counts')
            spk_mean_rate.append(np.squeeze(np.asarray(histogram_count)))
        spk_mean_rate = np.array(spk_mean_rate)
        
        corr_mat = np.corrcoef(spk_mean_rate)
        #take care of nans
        corr_mat[np.isnan(corr_mat)]=0
        
        
        #time_cell.append(shuffled_train)
               
        return corr_mat
    
        
    
    def parallel_corr(self, shuffled_train, pca_bin):
        
       
        timelapse_shuffled_corr = self.shuffle_preserve_isi_corr_control(shuffled_train, pca_bin)
        return timelapse_shuffled_corr


    def make_symmat_from_sttc(self, neotrain, combi, paral_result):
        #make matrix 
        sttc_mat = np.zeros((len(neotrain),len(neotrain))) 
        combi_idx=tuple(np.array(combi).T)
        sttc_mat[combi_idx]=np.array(paral_result)
        
        #symmetrize
        sym_mat = sttc_mat + sttc_mat.T - np.identity(len(neotrain))*np.diag(sttc_mat)
        
        return sym_mat
    
    def get_sttc_mat(self, shuf_trains, pca_bin=20, n_cpu=16):

       
        neotrain = self.to_neotrain()
            
        # combinatorics 
        
        combi = combinations(np.arange(len(neotrain)), 2)
        combi = list(combi)
        
        paral_result = self.parallel_sttc(neotrain, combi, pca_bin/2*pq.ms, n_cpu) #symmetric measure
        sym_mat = self.make_symmat_from_sttc(neotrain, combi, paral_result)
        
        sym_mat[np.isnan(sym_mat)]=0
        
        # compare to shuffled trains
        
        shuf_col =[]
        min_cell_shuf = []
        
                
        for shuffled in shuf_trains:
           
            shuffled_sttc = self.parallel_sttc(shuffled, combi, pca_bin/2*pq.ms, n_cpu) #symmetric measure
            sh_sym_mat = self.make_symmat_from_sttc(shuffled, combi, shuffled_sttc)
            shuf_col.append(sh_sym_mat)
            min_cell_shuf.append(np.min(sh_sym_mat))
            #testing 
        #min_cell_shuf_tp.append(np.array(min_cell_shuf))
        shuf_col = np.array(shuf_col)
        sort_col = np.sort(shuf_col, axis=0)
        
        test_mat = sym_mat - sort_col[-1,:,:] # test significance value
        eff_sttc_mat = np.copy(sym_mat)
        eff_sttc_mat[test_mat<0]=0
            
                    
        sttc_mat_result = dict()
        sttc_mat_result['raw_sttc_mat']=sym_mat
        sttc_mat_result['eff_sttc_mat']=eff_sttc_mat
        sttc_mat_result['sttc_shuffled_mins']=np.array(min_cell_shuf)
        
            
        return sttc_mat_result
    
    def parallel_sttc(self, neotrain, combi, dt, n_cpus):
        result = parmap.map(self.compute_sttc, combi, neotrain, dt, pm_processes=n_cpus, pm_pbar = False)
        
        return result
    
    def compute_sttc(self,combi,neotrain, dt):
    
        spiketrain1 = neotrain[combi[0]]
        spiketrain2 = neotrain[combi[1]]
        
        return spike_time_tiling_coefficient(spiketrain1, spiketrain2, dt)

