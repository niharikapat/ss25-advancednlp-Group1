Group 1
| Student ID | Member Name                  | Role                                                                                   |
|------------|------------------------------|----------------------------------------------------------------------------------------|
| 298762     | Maximilian Franz             | Paper: Why is this an important contribution to research and practice                 |
| 376365     | Upanishadh Prabhakar Iyer    | Paper: The research question addressed in the paper (thus, its objective)             |
| 371696     | Lalitha Kakara               | Paper: What are their results and conclusions drawn from it? What was new in this paper at the time of publication |
| 370280     | Muhammad Tahseen Khan        | Paper: What did the authors actually do (procedure, algorithms used, input/output data, etc) |
| 372268     | Dina Mohamed                 | Paper: What did the authors actually do (procedure, algorithms used, input/output data, etc) Model: Implemented live sentiment analysis in transformer & structured repo |
| 368717     | Yash Bhavneshbhai Pathak     | Model: DAN-based Encoder algorithm implementation                                     |
| 376419     | Niharika Patil               | Model: Transformer-based Encoder algorithm implementation                             |
| 373575     | Mona Pourtabarestani         | Paper: What are their results and conclusions drawn from it? What was new in this paper at the time of publication |
| 350635     | Divya Bharathi Srinivasan    | Model: DAN-based Encoder algorithm implementation                                     |
| 364131     | Siddu Vathar                 | Paper: Why is this an important contribution to research and practice                 |

# ss25-advancednlp

## Encoders

This project includes **two different sentence encoders** for binary sentiment classification:

1. **DAN** (Deep Averaging Network) — a simple and fast baseline model located in the `dan/` folder.
2. **Transformer-based encoder** — a more powerful contextual model located in the `transformer/` folder.

Each encoder is implemented independently and can be run separately. Refer to the `README.md` file inside each subfolder for detailed instructions on training, evaluation, and prediction.

## Dataset

The dataset used for both models comes from the **UCI Sentiment Labelled Sentences Data Set** and is located in the `/data/` folder. It includes:

- `amazon_cells_labelled.txt`
- `imdb_labelled.txt`
- `yelp_labelled.txt`

**Format:**
```
<sentence> \t <label>
```
- `label` is either `0` (Negative) or `1` (Positive)
