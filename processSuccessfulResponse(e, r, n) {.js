processSuccessfulResponse(e, r, n) {
    let i = [];
    for await (let a of e.chatCompletions) {
        this.accessor.get(Qr).sendTelemetryEvent(
            "response.success", {
                reason: a.finishReason,
                source: n?.messageSource ?? "unknown",
                requestId: r
            }, {
                tokenCount: a.numTokens,
                processingTime: e.getProcessingTime()
            }
        );
        this.isRepetitive(a, n) || i.push(a);
    }
    this.logger.debug(`Received choices: ${JSON.stringify(i, null, 2)}`);
    let o = i.filter(a => a.finishReason === "stop" || a.finishReason === "client-trimmed");
    if (o.length >= 1) {
        return {
            type: "success",
            value: o.map(a => a.message.content),
            requestId: r
        };
    }
    switch (i[0]?.finishReason) {
        case "content_filter":
            return {
                type: "filtered",
                reason: "Response got filtered.",
                requestId: r
            };
        case "length":
            return {
                type: "length",
                reason: "Response too long.",
                requestId: r
            }
    }
    return {
        type: "unknown",
        reason: "Response contained no choices.",
        requestId: r
    }
}

async getOffTopicModelName() {
    let e = Gt(this.accessor, Me.DebugOverrideChatOffTopicModel, {
        default: ""
    });
    if (e) {
        return e;
    }
    let r = await this.accessor.get(rn).chatOffTopicModel();
    return r || "";
}

async getOffTopicModelTokenizer() {
    let e = Gt(this.accessor, Me.DebugOverrideChatOffTopicModelTokenizer, {
        default: ""
    });
    if (e) {
        return e;
    }
    let r = await this.accessor.get(rn).chatOffTopicModelTokenizer();
    return r || "";
}

async getOffTopicModelThreshold() {
    let e = Gt(this.accessor, Me.DebugOverrideChatOffTopicModelThreshold, {
        default: 0
    });
    if (e !== 0) {
        return e;
    }
    let r = await this.accessor.get(rn).chatOffTopicModelThreshold();
    return r !== 0 ? r : 0;
}

isRepetitive(e, r) {
    let n = sX(e.tokens);
    if (n) {
        let i = Nt.createAndMarkAsIssued();
        i.extendWithRequestId(e.requestId);
        let o = i.extendedBy(r);
        this.accessor.get(rt).sendRestrictedTelemetry("conversation.repetition.detected", o);
        this.logger.info("Filtered out repetitive conversation result")
    }
    return n
}

processCanceledResponse(e, r) {
    return this.logger.debug("Cancelled after awaiting fetchConversation"), {
        type: "canceled",
        reason: e.reason,
        requestId: r
    }
}

processFailedResponse(e, r) {
    return e.failKind === "rateLimited" ? {
        type: "rateLimited",
        reason: e.reason,
        requestId: r
    } : e.failKind === "offTopic" ? {
        type: "offTopic",
        reason: e.reason,
        requestId: r
    } : e.failKind === "tokenExpiredOrInvalid" || e.failKind === "clientNotSupported" || e.reason.includes("Bad request: ") ? {
        type: "badRequest",
        reason: e.reason,
        requestId: r
    } : {
        type: "filtered",
        reason: e.reason,
        requestId: r
    }
}

processError(e, r) {
    return S0(e) ? {
        type: "canceled",
        reason: "network request aborted",
        requestId: r
    } : (this.logger.exception(e, "Error on conversation request"), e.code === "ENOTFOUND" ? {
        type: "failed",
        reason: "Network request failed. Please check your network connection and try again.",
        requestId: r
    } : {
        type: "failed",
        reason: "Error on conversation request. Check the log for more details.",
        requestId: r
    })
}

vX(t) {
    switch (t) {
        case 1:
            return "conversationInline";
        case 2:
            return "conversationPanel"
    }
}
