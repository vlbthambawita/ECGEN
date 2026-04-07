from ecgen import generate                                                                                                                                                                                                                                                     
                                         
  # Basic — generates CSV files with lead headers                                                                                                                                                                                                                                
generate(
      model_path="hf://vlbthambawita/ECGEN/pulse2pulse/ptbxl/pulse2pulse_exp_ptbxl_full_epoch:900.pt",
      n_samples=100,
      output_dir="outputs/",
      header=False,
      ecgplot=True,
  )                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                        
  # No headers                     
  #generate(..., header=False)                                                                                                                                                                                                                                                    
                                         
  # With ECG plot images  (pip install 'ecgen[plot]')                                                                                                                                                                                                                            
  #generate(..., ecgplot=True)                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                 
  #Output: one sample_001.csv … sample_N.csv per sample.                                                                                                                                                                                                                          
  # CSV format: 5000 rows × 8 columns — columns are I, II, V1, V2, V3, V4, V5, V6.     