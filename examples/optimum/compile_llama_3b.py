import os

from optimum.rbln import RBLNAutoModelForCausalLM


def main():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    model_dir = "./Llama-3.1-8b-b1-eager-4096-tp1"

    # Compile and export
    model = RBLNAutoModelForCausalLM.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=4096,  # default "max_position_embeddings"
        rbln_tensor_parallel_size=1,
    )

    # Save compiled results to disk
    model.save_pretrained(model_dir)


if __name__ == "__main__":
    main()