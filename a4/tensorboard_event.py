from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

ea = EventAccumulator('runs/nmt/events.out.tfevents.1786856929.autodl-pro-785510b969b0.1591.0',
                      size_guidance={'scalars': 0})
ea.Reload()
for tag in ['loss/train', 'perplexity/train', 'perplexity/val']:
    evs = ea.Scalars(tag)
    plt.plot([e.step for e in evs], [e.value for e in evs], label=tag)
plt.legend(); plt.xlabel('iter'); plt.ylabel('value')
plt.savefig('training_curves.png', dpi=150)