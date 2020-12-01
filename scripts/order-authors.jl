## Julia script to randomly decide the author of first authors
## Claudia (November 2020)

using Random
s = 313627913 ##decided by group in order of reply: zhaoyi (313), brian (627), claudia (913)
Random.seed!(s)
people = ["songyang","zhaoyi"] ##alphabetical
people[randperm(length(people))]