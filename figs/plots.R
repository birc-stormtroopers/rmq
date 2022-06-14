library(tidyverse)

growth <- tibble(
  b = seq(4, 16),
  exp = 2**(2*b),
  choose = choose(2*b,b),
  catalan = choose / (b + 1)
) 

growth |> pivot_longer(
  exp:catalan, names_to = "Growth", values_to = "Size"
) |> ggplot(aes(x = b, y = Size, color = Growth)) +
  geom_line() + geom_point() +
  scale_y_log10() +
  scale_color_discrete(
    name='', 
    labels=expression(C[b], bgroup("(", atop(2*b,b), ")"), 2**{2*b})
  ) +
  theme_light() + 
  theme(legend.position = "top")

growth |> ggplot(aes(x = b, y = catalan/exp)) +
  geom_line() + geom_point() +
  ylab(expression("Fraction of table used -- " * 2**{2*b}/C[b])) +
  theme_light()
