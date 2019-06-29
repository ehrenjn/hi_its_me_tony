import discord
import json
import datetime

bot = discord.Client()

def get_token():
    with open('bot_token.txt') as token_file:
        return token_file.read()

@bot.event
async def on_ready():
    memechat = bot.get_guild(338145800620212234)
    all_messages = []
    for channel in memechat.channels:
        if isinstance(channel, discord.TextChannel):
            print(f'checking: {channel.name}')
            try:
                async for message in channel.history(limit = None):
                    if not message.author.bot and message.webhook_id is None: #don't get bot or webhook posts
                        all_messages.insert(0, {
                            'content': message.content,
                            'not_text': message.type == discord.enums.MessageType.default,
                            'author': {
                                'is_bot': message.author.bot,
                                'name': message.author.name,
                                'discriminator': message.author.discriminator,
                                'id': str(message.author.id)
                            },
                            'has_embeds': any(message.embeds),
                            'has_mentions': any(message.mentions),
                            'has_attachments': any(message.attachments),
                            'has_reactions': any(message.reactions),
                            'id': str(message.id),
                            'pinned': message.pinned,
                            'raw_mentions': {
                                'users': message.raw_mentions,
                                'channels': message.raw_channel_mentions,
                                'roles': message.raw_role_mentions
                            },
                            'created_at': datetime.datetime.timestamp(message.created_at),
                            'channel': {
                                'name': channel.name,
                                'id': str(channel.id)
                            }
                        })
                        if len(all_messages) % 1000 == 0:
                            print(f'got {len(all_messages)} messages')
            except discord.errors.Forbidden:
                print("don't have permissions, skipping channel")
    print("converting to json...")
    messages_json = json.dumps(all_messages)
    print("dumping to file...")
    with open('all_messages.json', 'w') as messages_dump:
        messages_dump.write(messages_json)
    print("done!")
    exit()

bot.run(get_token())