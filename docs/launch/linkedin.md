# LinkedIn draft

---

I spent the last few weeks chasing a slightly ridiculous idea, and I want to share it honestly — including the part where it didn't work.

I have a MacBook M3 Pro with 18 GB of RAM. Like a lot of you, I kept running into the same ceiling: the models I actually wanted to run wouldn't fit in memory. So instead of buying a bigger machine, I asked a stubborn question. Does a language model really have to live in RAM to run at all? Or is that just how everyone happens to do it?

The idea behind rapidAI: modern Mixture-of-Experts models only use a small slice of themselves per token. So keep the always-on parts in memory and stream the rest off the SSD, on demand, only when a token actually needs them.

It worked further than I expected. A 30B model now runs comfortably on my laptop at a few words a second. And GPT-OSS-120B — a model nearly four times bigger than my entire RAM, one that normally won't even open on this machine — actually runs and produces coherent text. Slowly, at about a word a second, but it runs.

Here's the part I'm most proud of, and it's not the demo. My real goal was to make that 120B fast enough for chat, and I couldn't. So I did the disciplined thing: I ran a series of pre-registered experiments — each with a kill threshold written down *before* I looked at the result — and I mapped exactly where the wall is. It turned out to be physics, not sloppy code: a synchronization step that's baked into how sparse models fetch their weights. I even wrote a native C++/Metal version specifically to prove the wall wasn't my Python. Same wall.

So this is a negative-leaning result, and that's the whole point. Five honest "this doesn't work, here's the number" findings that will save the next person weeks. In a field that mostly publishes wins, I think the dead ends are worth more.

Everything's open source, MIT, negatives included. If you own any Apple Silicon Mac, I'd genuinely love for you to clone it, run the benchmark, and tell me your numbers — I want to see where this wall sits across hardware that isn't mine.

Repo, paper, and the full map in the comments. Always happy to talk shop.

#MachineLearning #LLM #AppleSilicon #OpenSource #MoE
</content>
