# iopublic

Stand alone version of the inferior olive model (without network generation & tuning)

Currently works with Arbor v0.6

## Example

```python
import iopublic
selected = '2021-12-08-shadow_averages_0.01_0.8_d1666304-c6fc-4346-a55d-a99b3aad55be'
iopublic.simulate_tuned_network(selected, tfinal=1, dt=0.025, gpu_id=0, spikes=())
```

##  Online version of 3-compartmental IO model

https://llandsmeer.github.io/IOJS/

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
