# Iris data attribution

Fisher, R. (1936). **Iris** [Dataset]. UCI Machine Learning Repository.
[DOI: 10.24432/C56C76](https://doi.org/10.24432/C56C76).

Source: [UCI dataset page](https://archive.ics.uci.edu/dataset/53/iris).
The source identifies R. A. Fisher as creator and licenses this dataset under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The dataset is supplied as is, without warranties; see the
[license and disclaimer](https://creativecommons.org/licenses/by/4.0/legalcode).
No endorsement by the creator, UCI, or Creative Commons is implied.

The bundled `iris.data` is the unchanged member of UCI's archive retrieved on
2026-10-23. Its original row order, duplicate feature records and the source's
noted differences from the Fisher article are retained. In particular, the
dataset page discusses discrepancies in records 35 and 38; this copy does not
silently correct them. The provenance JSON records URLs and SHA-256 hashes.

The loader only parses numbers and encodes the three class names as 0, 1, 2.
Each call returns new arrays. The capstone groups exact duplicate feature rows
before partitioning. These small, historical observations support a bounded
teaching benchmark, not a claim about modern deployed classification quality.

This data license applies to these dataset records and is distinct from any
license governing the repository's code or original educational prose.
