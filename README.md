## World model?

Current target:
- train world model simulate được multiplayer slither io (scaled down replication of [MIRA](https://mira-wm.com))
- world model nhiều ứng dụng phết, something else?


Task:
Cho frame[i-1], frame[i] và action[i -> i+n], hãy đoán frame[i+1 -> i+n] của game pong.
(cho 2 frame ban đầu bởi vì cần biết hướng bóng bay nữa)


Current sol:

Arch đơn giản nhất là: [DirectFramePredictor](src/pong_wm/models.py#L124)
features = [CNN_Encoder](src/pong_wm/models.py#L11)(concat(frame[t-1], frame[t])) (MLP không hợp cho ảnh 2d)
z = Linear(concat(features, action[t])) (ta không concat(frame[t-1], frame[t], action[t]) từ đầu được, bởi vì phải giữ ảnh 2d cho CNN; mà action là 1d)
frame[t+1] = [Decoder](src/pong_wm/models.py#L33)(z)

KQ:
![Direct pixel rollout](artifacts/direct_rollout.png)

Why bad: Not sure @@. Train thì loss thấp, nma loss != quality bởi vì objective t đặt penalize pixel có bóng / platform của reference image * 20 so với background, nên model hack bằng cách tô toàn bộ mấy ô đó. T đoán nguyên nhân là vì cặp encoder - decoder train trên task đoán frame tiếp theo cùng với MLP, chứ không phải là reconstruct trước, cho nên là lỏ.

Sửa:

1. Thay vì (frame[t-1], frame[t], action[t]) -> model -> (frame[t], frame[t+1], action[t+1]) -> model -> ...
   thì: (frame[t-1], frame[t]) -> encoder -> z[t] (latent chứa info của cả frame[t, t-1]);
   (z[t], action[t]) -> [WorldModel MLP](src/pong_wm/models.py#L78) -> (z[t+1], action[t+1]) -> MLP -> (z[t+2], action[t+2])...
   bây giờ ta đã có z[t+1 -> t+n] -> decode -> frame[t+1 -> t+n]
lợi:
- latent "z" chỉ có dimension = (64,); tức là đoán frame tiếp theo rẻ hơn hẳn so với project lại về pixel space (3, 210, 160) = 100,800.
- latent chỉ gồm "thông tin có ích" của frame, MLP giờ không cần đoán từng pixel thay đổi ntn mà chỉ cần đoán cái latent thay đổi ntn. (ở đây, latent chắc là chủ yếu chứa vị trí của platform, hướng bóng, bảng điểm).

2. freeze encoder với cái decoder; train [Autoencoder](src/pong_wm/models.py#L57) riêng trên task reconstruct bình thường, xong train cái MLP ở trong trên task world model sau. Bây giờ ta chắc chắn được là không có nhiều thông tin bị mất khi encoder nén frame thành latent.
![Autoencoder reconstruction](artifacts/autoencoder_result.png)

KQ world model sau khi sửa ([WorldModel](src/pong_wm/models.py#L78))
![World model rollout at horizon 16](artifacts/rollout_h16.png)

Nma, model vẫn không coherent quá 16 frame :(
![World model rollout at horizon 32](artifacts/rollout.png)


How to run:

```bash
# setup env
uv sync

# collect 20k train + 2k validation frames
uv run pong-wm collect

# train autoencoder trước, rồi mới train latent dynamics
uv run pong-wm train-ae
uv run pong-wm train-world

# validation loss + action ablation + artifacts/rollout.png
uv run pong-wm evaluate
```

Muốn chạy direct-pixel baseline ở trên:

```bash
uv run pong-wm train-direct
uv run pong-wm evaluate-direct
```
