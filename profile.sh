#/bin/bash
# Runs qii tool for a single individual and then runs the profiler on top level functions and their direct callees.
python -m cProfile -o /tmp/prof_tmp.bin -s cumtime  qii.py  -m shapley adult
python -c "import pstats; p = pstats.Stats('/tmp/prof_tmp.bin').sort_stats('cumulative'); p.print_stats('qii_lib'); p.print_callees(1.0, 'qii_lib')"