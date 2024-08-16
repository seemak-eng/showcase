import utils

#region TestCases

# Index: 'youtube-tech' (YouTube Encyclopedia)

query_text_1 = "YouTub is own by whom?" # Answer: YouTube is owned by Google.
query_text_2 = "On which year YouTube was purchased by google?" # Answer: YouTube was purchased by Google in 2006.
query_text_3 = 'Type of business?' # Answer: The type of business is a subsidiary.
query_text_4 = 'When is the YouTube founded?' # Answer: YouTube was founded on February 14, 2005.

#endregion

query_with_contexts = utils.build_prompt(query_text_1)
print("TEST (1) --->\n",query_with_contexts)

response = utils.generate_response(query_with_contexts)
print("TEST (2) --->\n",response)