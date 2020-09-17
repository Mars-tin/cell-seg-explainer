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
The loss on test dataset is: 0.6996369957923889 | The accuracy on test dataset is: 0.5 | Obtained in epoch 9
Marker mask: tensor([0.5052, 0.5387, 0.4765, 0.4890, 0.4811, 0.4423, 0.5285])
Node:  2 ; Label: 1.0
Related nodes: [2, 18, 29, 69, 91, 92, 96, 136, 154, 155, 172, 177, 187, 205, 218, 219, 244, 250, 268, 291, 294, 297]
Related edges: [(2, 18), (2, 91), (2, 92), (2, 136), (2, 177), (2, 187), (2, 205), (2, 268), (18, 2), (18, 91), (18, 92), (18, 136), (18, 172), (18, 177), (29, 96), (29, 154), (29, 177), (29, 205), (29, 218), (29, 219), (29, 250), (29, 291), (29, 294), (29, 297), (69, 92), (69, 136), (69, 172), (69, 244), (91, 2), (91, 18), (91, 92), (91, 96), (91, 136), (91, 155), (91, 172), (91, 205), (91, 268), (92, 2), (92, 18), (92, 69), (92, 91), (92, 136), (92, 172), (92, 205), (92, 268), (96, 29), (96, 91), (96, 154), (96, 155), (96, 205), (96, 219), (96, 268), (136, 2), (136, 18), (136, 69), (136, 91), (136, 92), (136, 172), (136, 244), (136, 268), (154, 29), (154, 96), (154, 155), (154, 205), (154, 219), (154, 291), (154, 294), (154, 297), (155, 91), (155, 96), (155, 154), (155, 205), (155, 219), (155, 268), (172, 18), (172, 69), (172, 91), (172, 92), (172, 136), (172, 244), (172, 268), (177, 2), (177, 18), (177, 29), (177, 187), (177, 205), (177, 218), (177, 250), (177, 291), (177, 294), (177, 297), (187, 2), (187, 177), (187, 218), (187, 250), (187, 291), (187, 297), (205, 2), (205, 29), (205, 91), (205, 92), (205, 96), (205, 154), (205, 155), (205, 177), (205, 219), (205, 268), (205, 297), (218, 29), (218, 177), (218, 187), (218, 250), (218, 291), (218, 294), (218, 297), (219, 29), (219, 96), (219, 154), (219, 155), (219, 205), (219, 294), (244, 69), (244, 136), (244, 172), (250, 29), (250, 177), (250, 187), (250, 218), (250, 291), (250, 294), (250, 297), (268, 2), (268, 91), (268, 92), (268, 96), (268, 136), (268, 155), (268, 172), (268, 205), (291, 29), (291, 154), (291, 177), (291, 187), (291, 218), (291, 250), (291, 294), (291, 297), (294, 29), (294, 154), (294, 177), (294, 218), (294, 219), (294, 250), (294, 291), (294, 297), (297, 29), (297, 154), (297, 177), (297, 187), (297, 205), (297, 218), (297, 250), (297, 291), (297, 294)]
Node: 2 ; Label: 1 ; Marker: tensor([3.0875, 3.9387, 3.6071, 1.9518, 2.1397, 3.3283, 2.5987])
Node: 18 ; Label: 0 ; Marker: tensor([3.2510, 4.0548, 3.9600, 2.5835, 2.7232, 4.1239, 2.8265])
Node: 29 ; Label: 1 ; Marker: tensor([5.0235, 3.5054, 3.3592, 1.4175, 5.2436, 1.5772, 4.9223])
Node: 69 ; Label: 1 ; Marker: tensor([1.1731, 2.5968, 3.9494, 2.8367, 2.9135, 2.5695, 4.1494])
Node: 91 ; Label: 1 ; Marker: tensor([3.1712, 3.0389, 3.6266, 1.4420, 2.4930, 3.8450, 2.3244])
Node: 92 ; Label: 1 ; Marker: tensor([2.0066, 5.0421, 3.0381, 2.4211, 1.3076, 3.7293, 3.6991])
Node: 96 ; Label: 0 ; Marker: tensor([5.4884, 4.6960, 3.1418, 4.8334, 3.3557, 2.5227, 3.4664])
Node: 136 ; Label: 1 ; Marker: tensor([3.6436, 3.6889, 3.2746, 2.3964, 3.7089, 3.4228, 0.1169])
Node: 154 ; Label: 0 ; Marker: tensor([2.5018, 2.5686, 2.7210, 3.5298, 2.2606, 2.6240, 0.6278])
Node: 155 ; Label: 1 ; Marker: tensor([1.6183, 2.8876, 3.8979, 3.2951, 1.9012, 1.5997, 3.1747])
Node: 172 ; Label: 1 ; Marker: tensor([1.1618, 3.6530, 2.8116, 1.8242, 3.2873, 2.9971, 2.9634])
Node: 177 ; Label: 1 ; Marker: tensor([2.2082, 2.7320, 2.5034, 4.3364, 2.8800, 3.4615, 2.9535])
Node: 187 ; Label: 1 ; Marker: tensor([4.3971, 4.4977, 3.5653, 1.2002, 1.8953, 3.4071, 2.3714])
Node: 205 ; Label: 0 ; Marker: tensor([3.9897, 3.5859, 4.1364, 3.6716, 2.0258, 1.3803, 3.5726])
Node: 218 ; Label: 1 ; Marker: tensor([1.4380, 4.0061, 2.9560, 4.9596, 3.9423, 0.9949, 3.7550])
Node: 219 ; Label: 0 ; Marker: tensor([1.6035, 2.2405, 2.7492, 2.9059, 3.3976, 1.9771, 1.8493])
Node: 244 ; Label: 1 ; Marker: tensor([1.9552, 3.7899, 4.1023, 2.3029, 3.2073, 3.7592, 3.1006])
Node: 250 ; Label: 1 ; Marker: tensor([3.7837, 4.9013, 2.4751, 3.2744, 1.9000, 2.5956, 2.2647])
Node: 268 ; Label: 0 ; Marker: tensor([3.5917, 2.5238, 1.2874, 3.6130, 3.1296, 1.5940, 4.1794])
Node: 291 ; Label: 1 ; Marker: tensor([2.6069, 3.2938, 2.1235, 4.1170, 2.7264, 2.9090, 1.1710])
Node: 294 ; Label: 1 ; Marker: tensor([6.8017, 5.3152, 3.1398, 4.7389, 2.9546, 2.9469, 1.0504])
Node: 297 ; Label: 1 ; Marker: tensor([2.7562, 2.8105, 3.4280, 3.5570, 1.2638, 2.6232, 2.0910])
```

The visualization is automatic:

![Sample Visualization](https://github.com/Mars-tin/cell-seg-explainer/blob/master/plot/sample.png)



## Label Explanation

TODO

