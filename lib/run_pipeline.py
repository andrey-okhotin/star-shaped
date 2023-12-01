import os
import sys
import subprocess

# run only from 'SS-DDPM' folder

def main():
    """
    CONSTRAINTS:
        Correctly work only from root of the repository!
    
    DESCRIPTION:
        Spawn processes for pytorch Distributed Data Parallel using 
        module subprocess and wait until they end. 
        For description of multiprocessing arguments goto
            - lib/pipelines/PipelinesRunner.py
        For description of pipelines arguments goto
            - lib/pipelines_configs/init_pipeline.py
    """
    
    # for disabling tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    list_of_gpus = sys.argv[2].split('_')
    num_processes = len(list_of_gpus)
    runner = os.path.join('lib', 'console_scripts', 'PipelinesRunner.py')
    program_name = 'python ' + runner + ' '
    processes = []
    print("", flush=True)
    for proc_id in range(num_processes):
        process_task = program_name + " ".join(sys.argv[1:]) + f" -nr {proc_id}"
        print(process_task, flush=True)
        processes.append(subprocess.Popen(process_task, shell=True))
    exit_codes = [ p.wait() for p in processes ]
    print(exit_codes, flush=True)

if __name__ == '__main__':    
    main()