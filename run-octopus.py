import os


graph = 'OctopusGraph'

attacks = [
   'none',
   'sign_flipping',
   'gaussian',
   'isolation',
   'sample_duplicate',
]

aggregations = [
   'ios',
   'faba',
   'cc',
   'scc',
]
partitions = [
   'noniid',
]

# action = os.system
action = print

# DSGD
for partition in partitions:
   for aggregation in aggregations:
      for attack in attacks:
         cmd = f'python "main DSGD.py" ' \
            + f'--graph {graph} ' \
            + f'--aggregation {aggregation} ' \
            + f'--attack {attack} ' \
            + f'--data-partition {partition} '
         action(cmd)
