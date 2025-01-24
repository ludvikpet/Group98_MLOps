import argparse
import random
import time

import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8080/predict")
    parser.add_argument("--wait_time", type=int, default=5)
    parser.add_argument("--max_iterations", type=int, default=1000)
    args = parser.parse_args()

    queries = [
        "I was anable to withdraw any money from my account.",
        "I tried to contact customer service but no one answered.",
        "I tried to use my card recently, but it was declined - what's going on?",
        "I noticed some suspicious activity on my account, and I believe it was hacked. Please help!",
        "I've entered my pin too many times, and now my card is blocked.",
        "I'd like to terminate my account, please.",
    ]

    negative_phrases = [
        "It’s getting frustrating to use.",
        "There are so many bugs now.",
        "The app crashes often and I am really disappointed.",
        "I think I'm going to stop using this app soon.",
        "It has become completely unusable.",
    ]

    count = 0
    while count < args.max_iterations:
        review = random.choice(queries)
        negativity_probability = min(count / args.max_iterations, 1.0)

        updated_review = review
        for phrase in negative_phrases:
            if random.random() < negativity_probability:
                updated_review += " " + phrase

        response = requests.post(args.url, json={"review": updated_review}, timeout=10)
        print(f"Iteration {count}, Sent review: {updated_review}, Response: {response.json()}")
        time.sleep(args.wait_time)
        count += 1