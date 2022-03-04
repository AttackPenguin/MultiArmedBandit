from scipy import stats

# dist = stats.expon()

# counter = 0
# for i in range(100):
#     value = float(dist.rvs() * 5)
#     print(value)
#     if value < 1:
#         counter += 1
#
# print(counter)

dist = stats.beta(.5, .01)
print(dist.mean())