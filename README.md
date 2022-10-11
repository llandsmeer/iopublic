# iopublic

Stand alone version of the inferior olive model (without network generation & tuning)

Currently works with Arbor v0.6

## Example

```python
import iopublic
selected = '2021-12-08-shadow_averages_0.01_0.8_d1666304-c6fc-4346-a55d-a99b3aad55be'
iopublic.simulate_tuned_network(selected, tfinal=1, dt=0.025, gpu_id=0, spikes=())
```
