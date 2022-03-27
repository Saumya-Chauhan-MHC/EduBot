# imports
import transformers
import torch
from transformers import pipeline, BertTokenizer, BertForNextSentencePrediction, AutoTokenizer, BertForMaskedLM, BertLMHeadModel
from torch.nn import functional as F
import discord
from dotenv import load_dotenv
import PyPDF2
import pdfplumber
from discord.ext import commands
from io import BytesIO
import os, asyncio, requests, json
import logging
import re
import numpy as np
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as analyzer


bot = commands.Bot(command_prefix=['!'], help_command=None)
dict = {}

# NLP/BERT set up for answering questions
unmask = pipeline('fill-mask', model='bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
context = ''
answer_question = pipeline("question-answering",model="bert-large-uncased-whole-word-masking-finetuned-squad")



# bot command for chat sentiment analysis
MSG_LIMIT = 100
@bot.command(name = 'chat_sentiment_analysis', pass_context=True)
async def chat_sentiment_analysis(ctx, *arg):

    amt = MSG_LIMIT

    has_mention = False
    has_integer = False
    has_channel = False
    many_ints = False

    mentions = ctx.message.mentions
    channel_mentions = ctx.message.channel_mentions

    sentence = []
    count_sen = 0

    if mentions:
        if len(mentions) > 1:
            await ctx.send("Incorrect number of mentions.")
            return
        has_mention = True

    if channel_mentions:
        if len(channel_mentions) > 1:
            await ctx.send("Incorrect number of channel mentions.")
            return
        has_channel = True


    for a in arg:
        try:
            amt = int(a)
            if has_integer:
                many_ints = True
                break
            has_integer = True
        except ValueError:
            pass

    if many_ints:
        await ctx.send("Incorrect number of integers: 1 maximum")
        return
    elif has_integer:
        print("Has an integer")

    current_channel_id = ctx.message.channel.id

    if has_channel:
        channel = ctx.message.channel_mentions[0]
    else:
        channel = bot.get_channel(current_channel_id)

    messages = await channel.history().flatten()

    for msg in messages:
        str_url_removed = re.sub('http[s]?://\S+', '', msg.content, flags=re.MULTILINE) # Remove urls
        str_mention_removed = re.sub('<@![0-9]+>', '', str_url_removed, flags=re.MULTILINE) # Remove mentions
        str_channel_removed = re.sub('<#[0-9]+>', '', str_mention_removed, flags=re.MULTILINE) # Remove channel
        if msg.author.id != bot.user.id and not '!' in str_channel_removed and str_channel_removed != '':
            sentence.append(str_channel_removed)
            count_sen += 1
        if has_mention and mentions[0].id == msg.author.id:
            sentence.append(str_channel_removed)
            count_sen += 1
        if count_sen == amt:
            break

    analyzer = analyzer()
    list = []
    for sen in sentence:
        sentence_dict = analyzer.polarity_scores(sen)
        list.append(sentence_dict['compound'])

    list_arr = np.array(list)
    mean = float(np.mean(list_arr))
    for spot,value in enumerate(list_arr):
        if list_arr[spot] < -0.1:
            list_arr[spot] = -1
        elif list_arr[spot] > 0.1:
            list_arr[spot] = 1
        else:
            list_arr[spot] = 0
    list_arr[spot] = value*((amt-spot)/amt)
    mean = float(np.mean(list_arr))


    percentage = (mean + 1) / 2 * 100

    summary_string = "The chosen " + str(amt) + " messages in " + channel.mention
    if has_mention:
        summary_string += " from " + mentions[0].mention
    summary_string += " had " + str(round(percentage , 3)) + "% positivity in the overall language."
    await ctx.send(summary_string)

# bot command for uploading PDF file
@bot.command()
async def uploadPDF(ctx):
    new_context = ""
    await ctx.message.channel.send("Uploading pdf file")
    attachment_url = ctx.message.attachments[0].url
    file_request = requests.get(attachment_url)
    raw_data = file_request.content
    with BytesIO(raw_data) as data:
        read_pdf = PyPDF2.PdfFileReader(data, strict=False)
        for page in range(read_pdf.getNumPages()):
            new_context = new_context + str(read_pdf.getPage(page).extractText())
            print(read_pdf.getPage(page).extractText())

    global context
    context = context + new_context
    await ctx.message.channel.send("PDF File Uploaded Successfully")


def get_quotes():
    response = requests.get('https://zenquotes.io/api/random')
    json_data = json.loads(response.text)
    quote = json_data[0]['q'] + ' - ' + json_data[0]['a']
    return quote

def printAssignments(tdl):
    assignment=""
    for index in tdl:
        assignment=assignment+'{ '+str(index.tasknum)+' : '+index.taskname+' - '+str(index.tasktime)+'mins }'+'\n'
    return assignment

class AssignmentEntry:
    def __init__(self,num,name,time):
        self.tasknum=num
        self.taskname=name
        self.tasktime=time


@bot.event
async def on_ready():
    print('Logged in as {0.user.name} ID: {0.user.id}'.format(bot))
    activity = discord.Game(name='S', type=0)
    await bot.change_presence(activity=activity)
    author=''
    try:
        with open('config.txt', 'r') as f:
            for line in f:
                if ':' not in line:
                    line = line[:-1]
                    dict[line]=[]
                    author=line
                    continue
                temp_list=line.split(':')
                if temp_list[2][-1] == '\n':
                    temp_list[2] = temp_list[2][:-1]
                dict[author].append(AssignmentEntry(int(temp_list[0]),temp_list[1],int(temp_list[2])))
    except Exception as e:
        print(e)

@bot.event
async def on_command_error(message, error):
    if isinstance(error, commands.BadArgument):
        await message.channel.send('Please enter a command with the correct arguments.')
    if isinstance(error, commands.CommandNotFound):
        await message.channel.send('Command is not found.')

# bot command for getting ping
@bot.command()
async def ping(message):
    await message.channel.send(f'My ping is {round(bot.latency*1000)}ms')

# bot command for viewing to-do list
@bot.command()
async def view_tdl(message):
    author=str(message.author)
    if not author in dict:
        await message.channel.send('Assignments List is Empty!')
    elif not dict[author]:
        await message.channel.send('Assignments List is Empty!')
    else:
        author=str(message.author)
        await message.channel.send(f"Assignments list of {message.author.mention}:")
        await message.channel.send(printAssignments(dict[author]))

# bot command for adding to to-do list
@bot.command()
async def todo(message, work: str='Generic', work_time: int=10):
    author=str(message.author)
    if author in dict:
        todolistx=dict[author]
    else:
        todolistx=[]
        dict[author]=todolistx
    todolistx.append(AssignmentEntry(len(todolistx)+1, work,work_time))
    await message.channel.send('Assignment is Saved!')

    try:
        with open('config.txt', 'w') as f:
            for i in dict:
                f.write(str(i)+'\n')
                for j in dict[i]:
                    f.write(f'{j.tasknum}:{j.taskname}:{j.tasktime}\n')
    except Exception as e:
        print(e)

# bot command for deleting/finishing task
@bot.command(aliases=['del'])
async def done(ctx, task_num: int):
    author=str(ctx.author)
    if not author in dict:
        await ctx.channel.send('Assignments List is Empty!')
    elif not dict[author]:
        await ctx.channel.send('Assignments List is Empty!')
    else:
        if task_num > 0:
            del dict[author][task_num-1]
            for i in range(task_num-1,len(dict[author])):
                dict[author][i].tasknum-=1

            await ctx.channel.send(f'Deleted task {task_num}\n')

            try:
                with open('config.txt', 'w') as f:
                    for i in dict:
                        f.write(str(i)+'\n')
                        for j in dict[i]:
                            f.write(f'{j.tasknum}:{j.taskname}:{j.tasktime}\n')
            except Exception as e:
                print(e)
        else:
            await ctx.channel.send('Please enter the correct task number.')

# bot command for starting task
@bot.command(aliases=['start'])
async def doing(message, task_number: int):
    author=str(message.author)
    if not author in dict:
        await message.channel.send('Assignments List is Empty!')
    elif not dict[author]:
        await message.channel.send('All your Tasks are finished!')
    else:
        if task_number > 0:
            time = dict[author][task_number-1].tasktime*60
            await message.channel.send('Task Started')
            await asyncio.sleep(time)
            await message.channel.send('Congratulations,{message.author.mention} you have completed task {task_num}!')

            quote = get_quotes()
            await message.channel.send(quote)

            if task_number > 0:
                author=str(message.author)
                del dict[author][task_number-1]
                for i in range(task_number-1,len(dict[author])):
                    dict[author][i].tasknum-=1

                await message.channel.send(f'Done task {task_number}\n')

                try:
                    with open('config.txt', 'w') as f:
                        for i in dict:
                            f.write(str(i)+'\n')
                            for j in dict[i]:
                                f.write(f'{j.tasknum}:{j.taskname}:{j.tasktime}\n')
                except Exception as e:
                    print(e)
            else:
                await message.channel.send('Please enter the correct task number.')
        else:
            await message.channel.send('Please enter the correct task number.')

# bot command for viewing list of commands (help)
@bot.command()
async def help(message):
    help_message='''```Command List\n
        prefix - !\n
        ping - get bot ping\n
        view_tdl - view your assignments\n
        todo [assignment number] [assignment time (in mins)] - add task to your list\n
        del [assignment number] - remove the task\n
        start [assignment number] - start the task\n
        chat_sentiment_analysis [number of messages] [user ID] [channel] - tone of classroom dialogue\n
        chat_sentiment_analysis - DEFAULT\n
        uploadPDF [PDF attachment] - add text data \n
        ```'''
    await message.channel.send(help_message)


# function for answering a question
def ansq(q1):
    #q1 = "What is the second unit?"
    print(q1)
    result1 = answer_question(question=q1, context=context)
    print("\n", q1)
    print(f"Answer: '{result1['answer']}', score: {round(result1['score'], 4)}, start: {result1['start']}, end: {result1['end']}")
    return result1['answer']

# bot command for answering question
@bot.command()
async def question(message, *, ques):
    await message.channel.send(ansq(ques))

load_dotenv()
TOKEN = os.getenv('TOKEN')
client = discord.Client()
bot.run(TOKEN)
