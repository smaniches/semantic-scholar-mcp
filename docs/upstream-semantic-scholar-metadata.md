# Semantic Scholar source-record observation: AlexNet

This note documents a publication-version discrepancy observed in a
Semantic Scholar Academic Graph response. It records the fields returned for
one paper and compares them with the relevant publication records.

## Record inspected

- **Observed:** November 12, 2026
- **Requested identifier:** `DOI:10.1145/3065386`
- **Semantic Scholar paper ID:** `abd1c342495432171beb7ca8fd9551ef13cbd0ff`
- **Paper title:** *ImageNet Classification with Deep Convolutional Neural Networks*

The API response included these values:

| Field | Value returned by Semantic Scholar |
| --- | --- |
| `year` | `2012` |
| `publicationDate` | `2012-12-03` |
| `externalIds.DBLP` | `conf/nips/KrizhevskySH12` |
| `venue` | `Communications of the ACM` |
| `journal` | Volume `60`, pages `84 - 90` |
| `externalIds.DOI` | `10.1145/3065386` |
| `citationStyles.bibtex` | ACM journal details with `year = {2012}`; no DOI field |

## Comparison with publication records

The [2012 NeurIPS proceedings](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c-Abstract.html)
contain the original conference paper. The [ACM publication](https://doi.org/10.1145/3065386)
identifies the later *Communications of the ACM* version as **2017**,
volume **60**, issue **6**, pages **84–90**.

The inspected Semantic Scholar response therefore pairs a 2012 year with
bibliographic details of the 2017 ACM publication. These values were present
in the upstream response before any processing by the MCP server.

## Relevance to this server

In default `bibtex` mode, the server returns Semantic Scholar's generated
entry unchanged. In optional `bibtex+doi` mode, it can add the DOI from the
same Semantic Scholar record. Neither mode chooses between publication
versions or corrects the year and other bibliographic fields.

This example describes one record as observed on the stated date. It does
not establish why Semantic Scholar associated these fields or how often
similar discrepancies occur. The provider's record may change. When a
version-specific citation matters, check the relevant proceedings or
publisher record.

[Semantic Scholar paper record](https://www.semanticscholar.org/paper/abd1c342495432171beb7ca8fd9551ef13cbd0ff)
