


# train method test code:
dttm_start = pd.Timestamp.now()
print(dttm_start)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters()
)
model.to(device)
train(model, loss_fn, optimizer,
      batch_size=64, training_rounds=1_000_000,
      report_modulus=200)

dttm_finish = pd.Timestamp.now()
print(dttm_finish)
print(dttm_finish-dttm_start)

model.train(False)

reward_totals = list()
for _ in range(1_000_000):
    gen = RewardGenerator()
    _, _, rewards = model([gen])
    reward_totals.append(sum(rewards[0]))
    # print(sum(rewards[0]))

print(np.mean(reward_totals))
