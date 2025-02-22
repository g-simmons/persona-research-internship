import cProfile
import io
import pstats
import os

#import functions to profile
#from ontology_scores import get_mats
import store_matrices

def profile_function(func, *args, **kwargs):
    output_dir = 'profiling_outputs'
    os.makedirs(output_dir, exist_ok=True)
    pr = cProfile.Profile()
    pr.enable()
    func(*args, **kwargs)
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    output_file = os.path.join(output_dir, 'profile_output_new.txt')
    with open(output_file, 'a') as f:
        f.write(f'{func.__name__}\n\n')
        f.write(s.getvalue())
    return 

if __name__ == "__main__":

    #profile_function(get_mats, "7B", "step1000", False, "olmo")
    profile_function(store_matrices.run_single_step)
    #store_matrices.run_single_step()