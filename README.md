# iopublic

Stand alone version of the inferior olive model (without network generation & tuning)

## Example

Run a simulation:

```python
import iopublic
selected = '2021-12-08-shadow_averages_0.01_0.8_d1666304-c6fc-4346-a55d-a99b3aad55be'
time, vs = iopublic.simulate_tuned_network(selected, tfinal=1, dt=0.025, gpu_id=0, spikes=())
```

Get voltage traces of all cells connected to a certain `gid`

```python
gid = ?
network = iopublic.get_network_for_tuning(selected)
idmap = {neuron.old_id:new_id for new_id, neuron in enumerate(network.neurons)}
neighbours = [idmap[trace.global_to] for trace in network.neurons[gid].traces]
# neighbours = [786, 709, 514, 514, 526, 831, 342, 173, 454, 633]
print(vs[neighbours])
```
