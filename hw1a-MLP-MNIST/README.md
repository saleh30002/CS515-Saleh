# HW1a: MNIST with MLP

In order to run the codes in this script, you can use one of the following two commands:

This one runs a particular experiment. An experiment corresponds to a particular MLP configuration that I used in this assignment. The details of each configuration can be found in the submitted report and also in the `parameter.py` file.

You can also find the names of the experiments to run below for ease of access.

```
python main.py --experiment <experiment_name> --mode <training / testing / both> --device <cpu / gpu>
```

This one runs all experiments that are listed in `parameter.py` file, one after the other. There are 30 experiments so this can take a very long time.

```
python main.py --runall --mode <training / testing / both> --device <cpu / gpu>
```

### Experiments

As mentioned before, each experiement is an MLP configuration. Please refer to the report or to `parameters.py` for details.
