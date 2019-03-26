from gensim.summarization import summarize, keywords
import pandas


data_frame = pandas.read_csv("datasets/articles1.csv")
only_content_frame = data_frame["content"]

# scratch = open("scratch.txt", 'w')
# print(only_content_frame[0], file=scratch)
# scratch.close()

article1 = only_content_frame[0]
summary1 = summarize(summarize(article1))

summary2 = "Collyer ruled that House Republicans had the standing to sue " \
           "the executive branch over a spending dispute and that the Obama " \
           "administration had been distributing the health insurance subsidies, " \
           "in violation of the Constitution. Even the congress didn't approve for the same."

# print(summarize(summary1))
print(summarize(summary2))
# print(summary1)
