FROM ubuntu:focal

ENV DENO_VERSION=1.0.5

RUN apt update \
    && apt upgrade -y \
    && apt -qq install -y ca-certificates curl unzip --no-install-recommends \
    && curl -fsSL https://github.com/denoland/deno/releases/download/v${DENO_VERSION}/deno-x86_64-unknown-linux-gnu.zip \
    --output deno.zip \
    && unzip deno.zip \
    && rm deno.zip \
    && chmod 777 deno \
    && mv deno /usr/bin/deno \
    && apt -qq remove -y ca-certificates curl unzip \
    && apt -qq -y autoremove \
    && apt -qq clean \
    && rm -rf /var/lib/apt/lists* /tmp/* /var/tmp/*

ADD . /app/
WORKDIR /app

RUN deno cache app.ts
EXPOSE 8080

CMD [ "deno", "run", "--allow-net", "app.ts" ]