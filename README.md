# DS312

## Train

```
python main.py train
    --root_path={ path folder default ./}
    --batch_size={ num batch_size default 4}
    --num_epochs={num num_epochs default 16}
    --lr={ num lr default 1e-5}
    --log_wandb={true or false is want use wandb  }
    --load_weights={true or false is load weight trained}
    --path_weights={path folder weight path  weight default ./}
```

Lưu ý tên filde weight sẽ đc lưu tự động có thể sửa trong code, nó sẽ được lưu tại path_weights. path_weights vừa là file load vừa là file lưu để thực hiện train liên tục.

## Predict

```
python main.py predict
    --root_path={ path folder default ./}
    --path_weights={path file weight path  weight default }
```

Lưu ý path_weights ở predict sẽ là path của file wight.
Đầu ra của predict sẽ là 2 file run.csv và valid.csv lần lượt là caption dự đoạn của tập test và tập valid
