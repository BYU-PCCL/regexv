# regexv
Regex using word embeddings for text matching


## Usage

```python
import regexv as re

string_to_search = "..."

regex = r'Your mother was a hamster and your father smelt of <elderberries,strawberries>'

print(re.search(regex, string_to_search))
```

and `regexv` will search for any word similar to: `<elderberries,strawberries>` (all berry like fruits).

So these sentences would all be positive matches:
- `Your mother was a hamster and your father smelt of cherries`
- `Your mother was a hamster and your father smelt of elderberries`
- `Your mother was a hamster and your father smelt of lingonberries`

Which saves you the hassle of writing an exhaustive regex like:
```python

regex = r'Your mother was a hamster and your father smelt of (elderberries|strawberries|cherries|lingonberries|huckleberries|mulberries|...)'

```