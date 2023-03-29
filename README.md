# Make Subscription using OpenAI



## Methodology

1. Prepare subtitle

    [video-watching-acceleration](https://github.com/studentofkyoto/video-watching-acceleration/) was used.
    
    this video-watching-acceleration use [Whisper by OpenAI](https://github.com/openai/whisper)(MIT license).

    this process makes transcript with video's language.

2. Translation to other language you want

    using ChatGPT API, subscription is translated to other language. 



## How to use

Install requirements

 `pip install -r requirements.txt`,

and locate mp4 file at this folder, Make subscript. (whisper model: large)

 `sh MakeSub.sh`

translate language you want.

`python Translation.py`






