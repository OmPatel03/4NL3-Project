# Worker Setup

This bundle is configured for **code submissions**, so a compute worker must be available on your Codabench instance.

## Recommended organizer steps

1. Create a dedicated queue for this competition in Codabench Queue Management.
2. Copy the queue broker URL or vhost from the queue details page.
3. Provision at least one Docker-enabled compute worker bound to that queue.
4. Update `competition.yaml` to set the competition `queue` field to the queue vhost before launch.
5. Upload the zipped competition bundle.

## Runtime choice

This bundle intentionally uses the default Codabench runtime image. The starter kit and scoring pipeline therefore avoid non-stdlib dependencies.

## Worker notes

- one CPU worker is enough for a classroom-sized competition
- add more workers if you expect concurrent submissions
- if you keep the `queue` field unset, Codabench uses the standard shared queue instead of a dedicated queue
