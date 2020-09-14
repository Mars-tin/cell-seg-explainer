# cell-seg-explainer

An GNN explainer on Synthetic dataset over cell segmentation data.

## Node Explanation

### Usage

```python
python explain_node.py -m gcn -n 1
```

* `-m` / `--model`: Name of the GNN model.
* `-n` / `–node`: Node index to explain.

### Example

Run `python explain_node.py` by default:

```bash
The loss on test dataset is: 0.7018610239028931 | The accuracy on test dataset is: 0.5 | Obtained in epoch 509
Explain node 0: 100%|██████████| 509/509 [00:00<00:00, 584.80it/s]
Node:  0 ; Label: 0
Related nodes: [0, 4, 35, 37, 81, 106, 131, 223, 233, 268, 290, 299]
Related edges: [(0, 106), (0, 223), (0, 233), (0, 268), (0, 290), (4, 81), (4, 223), (4, 299), (35, 37), (35, 81), (35, 106), (35, 131), (35, 223), (35, 233), (35, 268), (35, 290), (37, 35), (37, 81), (37, 131), (37, 223), (37, 233), (37, 268), (37, 290), (81, 4), (81, 35), (81, 37), (81, 131), (81, 223), (106, 0), (106, 35), (106, 223), (106, 233), (106, 268), (106, 290), (131, 35), (131, 37), (131, 81), (131, 233), (131, 268), (223, 0), (223, 4), (223, 35), (223, 37), (223, 81), (223, 106), (223, 233), (223, 268), (223, 290), (223, 299), (233, 0), (233, 35), (233, 37), (233, 106), (233, 131), (233, 223), (233, 268), (233, 290), (268, 0), (268, 35), (268, 37), (268, 106), (268, 131), (268, 223), (268, 233), (268, 290), (290, 0), (290, 35), (290, 37), (290, 106), (290, 223), (290, 233), (290, 268), (299, 4), (299, 223)]
Marker importance: tensor([0.0444, 0.0429, 0.0413, 0.0556, 0.0417, 0.0410, 0.0396])
Node: 0 ; Label: 0 ; Marker: tensor([2.7451, 2.8591, 2.3381, 3.2590, 3.0144, 1.5204, 2.7593])
Node: 4 ; Label: 1 ; Marker: tensor([4.3880, 3.2262, 3.5691, 3.1973, 2.8136, 2.6448, 3.0961])
Node: 35 ; Label: 0 ; Marker: tensor([4.2177, 1.6166, 3.7174, 1.7523, 4.4623, 3.5166, 2.7426])
Node: 37 ; Label: 1 ; Marker: tensor([1.6887, 4.0041, 3.8730, 4.3941, 2.4112, 3.1862, 3.8583])
Node: 81 ; Label: 0 ; Marker: tensor([2.0014, 3.5575, 2.5573, 3.0811, 6.4327, 2.2525, 2.4133])
Node: 106 ; Label: 1 ; Marker: tensor([3.1652, 3.9763, 2.3040, 4.9371, 4.3492, 1.7249, 4.0091])
Node: 131 ; Label: 0 ; Marker: tensor([3.5570, 2.8683, 5.1723, 4.1771, 4.6401, 2.5821, 2.1031])
Node: 223 ; Label: 1 ; Marker: tensor([3.5292, 3.1654, 4.2445, 3.3746, 4.4513, 4.0724, 3.3590])
Node: 233 ; Label: 0 ; Marker: tensor([2.2046, 3.4837, 1.6772, 3.5403, 2.4927, 4.1284, 1.7766])
Node: 268 ; Label: 0 ; Marker: tensor([4.8830, 3.7849, 2.3153, 3.0423, 3.3075, 2.9100, 2.0362])
Node: 290 ; Label: 0 ; Marker: tensor([1.8556, 3.4847, 3.3145, 4.6109, 4.0559, 1.8817, 2.2228])
Node: 299 ; Label: 0 ; Marker: tensor([2.0772, 1.7995, 3.5825, 5.0435, 3.3431, 4.8198, 3.3380])
```

The visualization is automatic:

![Sample Visualization](https://github.com/Mars-tin/cell-seg-explainer/blob/master/plot/sample.png)



## Label Explanation

TODO

