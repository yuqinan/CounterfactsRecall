from rich.table import Table
import rich
import matplotlib.pyplot as plt
import numpy as np

def print_attn_map(attentions, input_tokens=None, head_idx=0, compress=True, colormap='summer', console=None):
    """
    attentions: torch.tensor. the attention weights output from a huggingface model e.g., when output_attentions=True is passed to the forward function
                for a single example. Has shape (num_heads, sequence_length, sequence_length)
    input tokens: an iterable of strings with length==sequence_length. If None, will just use a range of numbers the length of the sequence
    head_idx: which head to visualize. Pass head_idx=None to visualize all of them sequentially
    compress: helpful for longer sequences in terminal. Vertical tokens and shows only color intensity and the tenths place (unless score is 1.0, then shows 1)
    colormap: a string corresponding to a matplotlib colormap or a function which returns an rgb color (with each channel in [0,1]) given a number in [0,1]
    console: A rich console object. optional. If passed, will use this to do all printing
    """
    assert len(attentions.shape)==3
    if head_idx == None or head_idx=='all':
        for i in range(len(attentions)):
            print_attn_map(attentions, input_tokens=input_tokens, head_idx=i, compress=compress, colormap=colormap, console=console,)
        return
    if type(colormap)==str:
        cmap = plt.get_cmap(colormap)
    else:
        cmap=colormap
    attentions = attentions.detach().cpu().numpy()
    attentions = attentions[head_idx] #index at the head
    if input_tokens == None:
        input_tokens = [str(i) for i in range(len(attentions))]
    table = Table(title=f"[b]Attention for Head no. {head_idx}[/]")
    table.add_column('SEQ:', justify='center')
    for tok in input_tokens:
        if compress:
            table.add_column('\n'.join(list(tok)), justify='center', no_wrap=False) #spell the tokens vertically to save horizontal space
        else:
            table.add_column(tok, justify='center', no_wrap=False)
    def format_row(row):
        new_row = []
        for r in row:
            color = [int(c*255) for c in cmap(r)]
            #print(color)
            color = f"rgb({color[0]},{color[1]},{color[2]})"
            if compress:
                new_row.append( f"[black on {color}]{str(r*10)[0]}[/]")#{str(r*10)[0]}
            else:
                new_row.append( f"[{color} on black]%5.4f[/]" % r)#
        return new_row
    for i,row in enumerate(attentions):
        new_row = [f"[b]{input_tokens[i]}[/]", *format_row(row)]
        table.add_row(*new_row)
    if console:
        console.print(table)
    else:
        rich.print(table)


def print_attn_map_row(attentions, row_idx, input_tokens=None, head_idx=0, sort=False, colormap='summer', console=None): #coolwarm is also a good colormap
    """
    attentions: torch.tensor. the attention weights output from a huggingface model e.g., when output_attentions=True is passed to the forward function
                for a single example. Has shape (num_heads, sequence_length, sequence_length)
    row_idx: the index of the token in the sequence that you want to see attentions FROM
    input tokens: an iterable of strings with length==sequence_length
    head_idx: which head to visualize. Pass head_idx=None to visualize all of them sequentially
    console: A rich console object. optional. If passed, will use this to do all printing
    """
    assert len(attentions.shape)==3
    if head_idx == None or head_idx == 'all':
        for i in range(len(attentions)):
            print_attn_map_row(attentions, row_idx, head_idx=i, input_tokens=input_tokens, colormap=colormap, console=console)
        return
    if type(colormap)==str:
        cmap = plt.get_cmap(colormap)
    else:
        cmap=colormap
    attentions = attentions.detach().cpu().numpy()
    print(attentions.shape)
    attentions = attentions[head_idx,:,row_idx] #index at the head, now a list of size (seq_len)
    #print(attentions[5])
    no_index=False
    if input_tokens == None:
        no_index=True
        input_tokens = [str(i) for i in range(len(attentions))]
    
    query = input_tokens[row_idx]
    table = Table(title=f"[b]Attention for Head no. {head_idx} to \"{query}\" (at index {row_idx})[/]")
    if not no_index:
        table.add_column("Index", justify='left')
    table.add_column(f'[b]From: "[green][/]", \nTo:{query}', justify='left', no_wrap=True)
    table.add_column("Attention Score:", justify="center", no_wrap=True)
    max_in_row = attentions.max()
    min_in_row = attentions.min()
    if sort:
        attn_idxs = np.argsort(attentions)[::-1]
    else:
        attn_idxs = range(len(attentions))
    def format_score(r):
        #intensity = int(r*255) # r in [0,1] this gives 'how colorful something is'
        #color = f"rgb(0,{intensity},0)"
        sat = (r-min_in_row)/(max_in_row-min_in_row)
        color = [int(c*255) for c in cmap(sat)]
        color = f"rgb({color[0]},{color[1]},{color[2]})"
        return f"[{color} on black]%5.4f[/]" % r
    def clean(token):
        if token == '\n':
            return '\\n'
        return token.strip('\n')
    for i in attn_idxs:#for i,score in enumerate(attentions):
        score = attentions[i]
        new_row = [str(i), f"[b]{clean(input_tokens[i])}[/]", format_score(score)]
        if no_index:
            new_row = new_row[1:]
        table.add_row(*new_row)
    new_row = ["avg", f"[b]{clean(input_tokens[i])}[/]", format_score(np.mean(attentions))]
    table.add_row(*new_row)
    
    if console:
        console.print(table)
    else:
        rich.print(table)

def print_attn_map_row_new(attentions, row_idx, input_tokens=None, head_idx=0, sort=False, colormap='summer', console=None): #coolwarm is also a good colormap
    """
    attentions: torch.tensor. the attention weights output from a huggingface model e.g., when output_attentions=True is passed to the forward function
                for a single example. Has shape (num_heads, sequence_length, sequence_length)
    row_idx: the index of the token in the sequence that you want to see attentions FROM
    input tokens: an iterable of strings with length==sequence_length
    head_idx: which head to visualize. Pass head_idx=None to visualize all of them sequentially
    console: A rich console object. optional. If passed, will use this to do all printing
    """
    assert len(attentions.shape)==3
    attentions_ = attentions.detach().cpu().numpy()
    all_head = []
    if head_idx == None or head_idx == 'all':
        for i in range(len(attentions)):
            attentions_new = attentions_[i,:,row_idx] #index at the head, now a list of size (seq_len)
            #print(attentions[5])
            no_index=False
            if input_tokens == None:
                no_index=True
                input_tokens = [str(i) for i in range(len(attentions_new))]
            query = input_tokens[row_idx]
            max_in_row = attentions_new.max()
            min_in_row = attentions_new.min()
            if sort:
                attn_idxs = np.argsort(attentions_new)[::-1]
            else:
                attn_idxs = range(len(attentions_new))
            print("head: {} ".format(i), np.mean(attentions_new))
            if np.mean(attentions_new) > 0.08:
                print(i)
            all_head.append(np.mean(attentions_new))
    return np.mean(all_head)



if __name__=="__main__":
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    inp = 'The capital of France is Beijing.\nQ: What is the capital of France?\nA:'
    print(inp)
    inp_ids = tokenizer.encode(inp, return_tensors='pt')

    tokens = [tokenizer.decode(i) for i in inp_ids[0]]

    out = model(input_ids=inp_ids, output_attentions=True)
    attns = out.attentions

    def redmap(r):
        #r is in [0,1]
        import colorsys
        return colorsys.hsv_to_rgb(1, r, 1)
    
    #print_attn_map(attns[17][0], input_tokens = tokens, head_idx=None, colormap=redmap) #layer 0, batch_idx 0, head 5
    #print_attn_map_row(attns[0][0], 17, input_tokens=tokens, head_idx=5)
    #only attention from ' is' to the other words
    l = []
    for i in range(12):
        print("****** layer {} ******".format(i))
        l.append(print_attn_map_row_new(attns[i][0], 5, input_tokens=tokens, head_idx=None))
    print(l) #row_idx = 4, index of ' is'


    def redmap(r):
        #r is in [0,1]
        import colorsys
        return colorsys.hsv_to_rgb(1, r, 1)

    #example with custom colormap
    #print_attn_map(attns[0][0], input_tokens = tokens, head_idx=9, colormap=redmap) #layer 0, batch_idx 0, head 9 (more variation here in gpt2)

