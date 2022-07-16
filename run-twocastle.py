import os


graph = 'TwoCastle'

attacks = [
   'none',
   'sign_flipping',
   'gaussian',
   'isolation',
   'sample_duplicate',
]

aggregations = [
   'mean',
   'ios',
   'trimmed-mean',
   'median',
   'geometric-median',
   'faba',
   'Krum',
   'cc',
   'scc',
]
partitions = [
   'iid',
   'noniid',
]

action = os.system
# action = print

# no communication
for partition in partitions:
   cmd = f'python "main DSGD.py" ' \
         + f'--graph {graph} ' \
         + f'--aggregation no-comm ' \
         + f'--attack none ' \
         + f'--data-partition {partition} '
   action(cmd)
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
# DRSA
for partition in partitions:
   for attack in attacks:
      cmd = f'python "main RSA.py" ' \
         + f'--graph {graph} ' \
         + f'--attack {attack} ' \
         + f'--data-partition {partition} '
      action(cmd)
