import argparse
import ollama
from rag import PROMPT, SYSTEMS, Retriever, cited_ids

def main():
    parser = argparse.ArgumentParser(description="Interactive RAG Chat")
    parser.add_argument("--system", choices=SYSTEMS, default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--index", default="index.npz")
    parser.add_argument("--chunks", default="corpus/chunks.json")
    args = parser.parse_args()

    retriever = Retriever(index=args.index, chunks=args.chunks)
    print(f"Loaded [{args.system}] | Model: {args.model} | Top-K: {args.top_k}\n" + "-"*50)

    while True:
        try:
            query = input("\nQuery: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query: continue
        if query.lower() in {"exit", "quit"}: break

        ranking = retriever.search(args.system, query, k=args.top_k)
        if not ranking:
            print("\nAnswer: Not in the documents.")
            continue

        context = "\n\n".join([f"[{cid}] {retriever.text[cid]}" for cid, _ in ranking])
        prompt = PROMPT.format(context=context, question=query)

        print("\nAnswer: ", end="", flush=True)
        full_ans = ""
        try:
            for chunk in ollama.chat(model=args.model, messages=[{"role": "user", "content": prompt}], stream=True):
                content = chunk["message"]["content"]
                print(content, end="", flush=True)
                full_ans += content
            print()
        except Exception as e:
            print(f"\n[Error: {e}]")
            continue

        cited = cited_ids(full_ans)
        if cited:
            print("\n--- Cited Source Snippets ---")
            for cid in cited:
                if cid in retriever.text:
                    print(f"  • [{cid}]: {retriever.text[cid]}")
        else:
            print("\n--- Top Retrieved ---")
            for rank, (cid, score) in enumerate(ranking[:5], 1):
                print(f"  {rank}. {cid} ({score:.4f})")

if __name__ == "__main__":
    main()