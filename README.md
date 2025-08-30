# mnist-adversarial

### Overview
- **Environment ID**: `mnist-adversarial`
- **Short description**: Evaluation environment for testing AI models' ability to distinguish adversarial examples from normal MNIST digits while correctly identifying the digit class.
- **Tags**: single-turn, test, eval, mnist, adversarial-example

### Datasets
- **Primary dataset(s)**: `wambosec/adversarial-mnist` - A dataset containing both normal and adversarial MNIST digit examples
- **Source links**: [Hugging Face Dataset](https://huggingface.co/datasets/wambosec/adversarial-mnist)
- **Split sizes**: Uses test split by default, with configurable sample size (default: 50 normal + 50 adversarial = 100 total examples)

### Task
- **Type**: single-turn
- **Parser**: Custom regex parser that extracts responses in `\boxed{adversarial_X}` or `\boxed{normal_X}` format
- **Rubric overview**: Dual scoring system with +0.5 points for correct adversarial/normal classification and +0.5 points for correct digit identification (0-9)

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval mnist-adversarial
```

Configure model and sampling:

```bash
uv run vf-eval mnist-adversarial -m gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7 -a '{"size": 30}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- Models receive flattened 784-element arrays representing 28×28 grayscale MNIST images

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_split` | str | `"test"` | Dataset split to use (train/test/validation) |
| `size` | int | `50` | Number of normal and adversarial examples each (total = 2×size) |

### Input Format
The model receives a flattened array of 784 grayscale values (0-255) representing a 28×28 MNIST digit image in row-major order. The system prompt instructs the model to classify the image as either adversarial or normal and identify the digit class.

### Expected Output Format
Models must respond with exactly one line in the format:
- `\boxed{adversarial_X}` for adversarial examples (where X is the digit 0-9)
- `\boxed{normal_X}` for normal examples (where X is the digit 0-9)

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward: 0.5 for correct adversarial/normal classification + 0.5 for correct digit identification (max: 1.0) |
| `accuracy` | Exact match on target answer |