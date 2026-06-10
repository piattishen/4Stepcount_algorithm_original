# 4Stepcount_algorithm_original

Unzip 4_algo_original_code.zip in the location you want.

upload pipeline_cluster_v3.py, adept_runner_cluster_v3.R, verisense_runner_cluster.R

run it in cluster by slrum
For new slrum:
```
sed -i 's/\r//' "/home/wang.yichen8/pipeline_algorithm_stepcount/slrum_4algo_merged_5.sh"
```

Then run it
```
sbatch "/home/wang.yichen8/pipeline_algorithm_stepcount/slrum_4algo_merged_5.sh"
```
